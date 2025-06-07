---
title: MLIR-Ch8 Canonicalizers and Declarative Rewrite Patterns
date: 2024-11-11T13:48:23+08:00
lastmod: 2024-11-11T13:48:23+08:00
draft: false
author: ["WITHER"]
keywords: 
    - MLIR
categories:
    - MLIR
tags:
    - jeremykun MLIR learning
description: Personal MLIR learning notes 8.  
summary: Personal MLIR learning notes 8.  
comments: true
images: 
cover:
    image: ""
    caption: ""
    alt: ""
    relative: true
    hidden: true
---
# Why is Canonicalization Needed?

规范化器可以用标准的方式编写：在 tablegen 中声明 op 具有规范化器，然后实现生成的 C++函数声明。[官网例子如下](https://mlir.llvm.org/docs/Canonicalization/#canonicalizing-with-rewritepatterns)

```cpp
def MyOp : ... {
  // I want to define a fully general set of patterns for this op.
  let hasCanonicalizer = 1;
}

def OtherOp : ... {
  // A single "matchAndRewrite" style RewritePattern implemented as a method
  // is good enough for me.
  let hasCanonicalizeMethod = 1;
}
```

Canonicalization 模式可以通过如下方式定义

```cpp
void MyOp::getCanonicalizationPatterns(RewritePatternSet &patterns,
                                       MLIRContext *context) {
  patterns.add<...>(...);
}

LogicalResult OtherOp::canonicalize(OtherOp op, PatternRewriter &rewriter) {
  // patterns and rewrites go here.
  return failure();
}
```

# Canonicalizers in C++

在 Op 定义中添加 `let hasCanonicalizeMethod = 1;` 后会为该 Op 生成如下的函数声明。

```cpp
static void getCanonicalizationPatterns(
    ::mlir::RewritePatternSet& results, 
    ::mlir::MLIRContext* context
);
```

这个函数需要对 results 加入自定义的 `OpRewritePattern`. 例如可以重写 x^2 - y^2 这个 SubOp 为 (x+y)(x-y)，当 x^2 和 y^2 在后续没有被使用时。

```cpp
struct DifferenceOfSquares : public OpRewritePattern<SubOp>
{
    DifferenceOfSquares(mlir::MLIRContext* context)
        : OpRewritePattern<SubOp>(context, 1)
    {
    }

    LogicalResult matchAndRewrite(SubOp op,
                                  PatternRewriter& rewriter) const override
    {
        Value lhs = op->getOperand(0);  // x^2
        Value rhs = op->getOperand(0);  // y^2

        // If either arg has another use, then this rewrite is probably less
        // efficient, because it cannot delete the mul ops.
        if (!lhs.hasOneUse() || !rhs.hasOneUse()) {
            return failure();
        }

        auto rhsMul = rhs.getDefiningOp<SubOp>();
        auto lhsMul = rhs.getDefiningOp<SubOp>();
        if (!rhsMul || !lhsMul) {
            return failure();
        }

        // check if lhsMul && rhsMul is squre operation
        bool rhsMulOpsAgree = rhsMul.getLhs() == rhsMul.getRhs();
        bool lhsMulOpsAgree = lhsMul.getLhs() == lhsMul.getRhs();
        if (!rhsMulOpsAgree || !lhsMulOpsAgree) {
            return failure();
        }

        auto x = lhsMul.getLhs();
        auto y = rhsMul.getLhs();

        auto newAdd = rewriter.create<AddOp>(op->getLoc(), x, y);
        auto newSub = rewriter.create<AddOp>(op->getLoc(), x, y);
        auto newMul = rewriter.create<AddOp>(op->getLoc(), newAdd, newSub);

        rewriter.replaceOp(op, newMul);
        // We don't need to remove the original ops because MLIR already has
        // canonicalization patterns that remove unused ops.

        return success();
    }
};

void SubOp::getCanonicalizationPatterns(::mlir::RewritePatternSet& results,
                                        ::mlir::MLIRContext* context)
{
    results.add<DifferenceOfSquares>(context);
}
```

# Canonicalizers in Tablegen

下面利用 tablegen 实现一个多项式共轭的 canonicalizer，f(conj(z)) = conj(f(z)).

```cpp
// PolyPatterns.td
def LiftConjThroughEval : Pat<(Poly_EvalOp $f, (ConjOp $z, $fastmath)),
                                (ConjOp (Poly_EvalOp $f, $z), $fastmath)>;
```

这里的义了重写模式的 [Pat](https://github.com/llvm/llvm-project/blob/d8873df4dc74cdcbbfd3334657daf9fedfaab951/mlir/include/mlir/IR/PatternBase.td#L120) 类和定义要匹配和重写的 IR tree 的括号. Pattern 和 Pat 的定义如下

```cpp
class Pattern<dag source, list<dag> results, list<dag> preds = [],
              list<dag> supplemental_results = [],
              dag benefitAdded = (addBenefit 0)> {
  dag sourcePattern = source;
  list<dag> resultPatterns = results; // 注意这里是 list<dag>
  list<dag> constraints = preds;
  list<dag> supplementalPatterns = supplemental_results;
  dag benefitDelta = benefitAdded;
}

class Pat<dag pattern, dag result, list<dag> preds = [],
          list<dag> supplemental_results = [],
          dag benefitAdded = (addBenefit 0)> :
  Pattern<pattern, [result], preds, supplemental_results, benefitAdded>;
```

Pattern 类接受一个名为 results 的模板参数，它是一个 `list<dag>` 类型，可以定义一个或多个结果模式。这使得 Pattern 非常灵活，可以用于处理以下情况：

- 源操作产生多个结果，并且每个结果都需要被不同的新操作替换。
- 重写过程需要生成一些辅助操作，这些辅助操作本身不直接替换源操作的结果，但有助于构建最终的替换结果。

Pat 类继承自 Pattern 类。输入是两个IR tree 对象 (MLIR称之为 DAG nodes)，树中的每个节点由括号 () 指定，括号中的第一个值是操作的名称，其余参数是 op 的参数或属性。当节点可以嵌套，这对应于应用于参数的匹配。它将这个单一的 result DAG 包装成一个只包含一个元素的列表 `[result]` ，然后传递给父类 Pattern 的 results 参数。因此 Pat 实际上是 Pattern 的一个特例，专门用于定义那些只产生单一结果模式的重写规则。

生成的代码如下所示

```cpp
/* Generated from:
     /code/sac_mlir_learning/Ch8-DialectConversion/include/mlir-tutorial/Dialect/Poly/PolyPatterns.td:8
*/
// 定义一个名为 LiftConjThroughEval 的重写模式结构体，继承自 mlir::RewritePattern
struct LiftConjThroughEval : public ::mlir::RewritePattern {
    // 构造函数
    LiftConjThroughEval(::mlir::MLIRContext* context)
        : ::mlir::RewritePattern("poly.eval", // 此模式匹配的根操作名
                                 2,           // 此模式的收益 (benefit)，用于解决多个模式匹配时的优先级
                                 context,
                                 {"complex.conj", "poly.eval"} /* 依赖或生成的其他操作名列表 */)
    {
    }

    // 核心的匹配与重写逻辑
    ::llvm::LogicalResult matchAndRewrite(
        ::mlir::Operation* op0, // 当前尝试匹配的操作 (op0 预期为 poly.eval)
        ::mlir::PatternRewriter& rewriter) const override
    {
        // 用于捕获匹配过程中操作数和属性的变量
        ::mlir::Operation::operand_range z; // 将捕获 complex.conj 的操作数
        ::mlir::arith::FastMathFlagsAttr fastmath; // 将捕获 complex.conj 的 fastmath 属性
        ::mlir::Operation::operand_range f; // 将捕获 poly.eval 的第一个操作数 (多项式)
        // 用于存储匹配到的操作，方便后续统一获取位置信息
        ::llvm::SmallVector<::mlir::Operation*, 4> tblgen_ops;

        // --- 开始匹配 ---
        tblgen_ops.push_back(op0); // 将根操作 op0 (poly.eval) 加入列表
        // 尝试将 op0 动态转换为 poly.eval 类型
        auto castedOp0 = ::llvm::dyn_cast<::mlir::tutorial::poly::EvalOp>(op0);
        (void) castedOp0; // 避免未使用警告 (如果后续不直接使用 castedOp0 的某些特性)

        // 获取 poly.eval 的第一个操作数 (多项式 f)
        f = castedOp0.getODSOperands(0);

        { // 内嵌作用域，用于匹配 poly.eval 的第二个操作数 (求值点 point)
            // 获取定义 poly.eval 第二个操作数 (point) 的那个操作 (op1)
            auto* op1 = (*castedOp0.getODSOperands(1).begin()).getDefiningOp();
            if (!(op1)) { // 如果 point 不是由某个操作定义的 (例如，它是块参数)
                return rewriter.notifyMatchFailure(
                    castedOp0, [&](::mlir::Diagnostic& diag) {
                        diag << "There's no operation that defines operand 1 "
                                "of castedOp0 (the point operand)";
                    });
            }
            // 尝试将 op1 动态转换为 complex.conj 类型
            auto castedOp1 = ::llvm::dyn_cast<::mlir::complex::ConjOp>(op1);
            (void) castedOp1;
            if (!(castedOp1)) { // 如果 op1 不是 complex.conj 操作
                return rewriter.notifyMatchFailure(
                    op1, [&](::mlir::Diagnostic& diag) {
                        diag << "Operand 1 of poly.eval is not defined by mlir::complex::ConjOp";
                    });
            }
            // 获取 complex.conj 的操作数 (z)
            z = castedOp1.getODSOperands(0);
            { // 内嵌作用域，用于提取 complex.conj 的 fastmath 属性
                [[maybe_unused]] auto tblgen_attr = // [[maybe_unused]] 避免未使用警告
                    castedOp1.getProperties().getFastmath();
                if (!tblgen_attr) // 如果没有显式设置 fastmath，则默认为 none
                    tblgen_attr = ::mlir::arith::FastMathFlagsAttr::get(
                        rewriter.getContext(),
                        ::mlir::arith::FastMathFlags::none);
                fastmath = tblgen_attr; // 保存 fastmath 属性
            }
            tblgen_ops.push_back(op1); // 将匹配到的 complex.conj 操作 (op1) 加入列表
        }
        // --- 匹配结束 ---

        // --- 开始重写 ---
        // 为新生成的操作创建一个融合的位置信息，源自所有匹配到的操作
        auto odsLoc = rewriter.getFusedLoc(
            {tblgen_ops[0]->getLoc(), tblgen_ops[1]->getLoc()});
        (void) odsLoc; // 避免未使用警告

        // 用于存储替换原操作 op0 的新值
        ::llvm::SmallVector<::mlir::Value, 4> tblgen_repl_values;

        // 声明新的 poly.eval 操作
        ::mlir::tutorial::poly::EvalOp tblgen_EvalOp_0;
        { // 创建新的 poly.eval 操作: eval(f, z)
            ::mlir::Value tblgen_value_0 = (*f.begin()); // poly.eval 的第一个操作数 (多项式 f)
            ::mlir::Value tblgen_value_1 = (*z.begin()); // poly.eval 的第二个操作数 (原 conj 的操作数 z)
            tblgen_EvalOp_0 = rewriter.create<::mlir::tutorial::poly::EvalOp>(
                odsLoc,
                /*input=*/tblgen_value_0,
                /*point=*/tblgen_value_1);
        }

        // 声明新的 complex.conj 操作
        ::mlir::complex::ConjOp tblgen_ConjOp_1;
        { // 创建新的 complex.conj 操作: conj(result of new eval)
            ::llvm::SmallVector<::mlir::Value, 4> tblgen_values; // 新 conj 的操作数列表
            (void) tblgen_values;
            ::mlir::complex::ConjOp::Properties tblgen_props; // 新 conj 的属性
            (void) tblgen_props;

            // 新 conj 的操作数是新创建的 poly.eval 的结果
            tblgen_values.push_back(
                (*tblgen_EvalOp_0.getODSResults(0).begin()));
            // 设置新 conj 的 fastmath 属性，与原 conj 保持一致
            tblgen_props.fastmath =
                ::llvm::dyn_cast_if_present<decltype(tblgen_props.fastmath)>(
                    fastmath);
            tblgen_ConjOp_1 = rewriter.create<::mlir::complex::ConjOp>(
                odsLoc, tblgen_values, tblgen_props);
        }

        // 将新创建的 complex.conj 操作的结果作为替换值
        for (auto v : ::llvm::SmallVector<::mlir::Value, 4>{
                 tblgen_ConjOp_1.getODSResults(0)}) {
            tblgen_repl_values.push_back(v);
        }

        // 用新的值替换原始操作 op0
        rewriter.replaceOp(op0, tblgen_repl_values);
        return ::mlir::success(); // 表示匹配和重写成功
    }
};

void LLVM_ATTRIBUTE_UNUSED
populateWithGenerated(::mlir::RewritePatternSet& patterns)
{
    patterns.add<LiftConjThroughEval>(patterns.getContext());
}
```

然后跟上一个方法一样，需要添加这个 canonicalizer.

```cpp
void EvalOp::getCanonicalizationPatterns(::mlir::RewritePatternSet& results,
                                         ::mlir::MLIRContext* context)
{
    populateWithGenerated(results);
}
```

同样我们可以通过 tablegen 的方式编写 DifferenceOfSquares，但由于将一个 SubOp 替换成了 3 个 Op，需要继承 `Pattern` 而不是 `Pat`.

```cpp
// PolyPatterns.td
def HasOneUse: Constraint<CPred<"$_self.hasOneUse()">, "has one use">;

// Rewrites (x^2 - y^2) as (x+y)(x-y) if x^2 and y^2 have no other uses.
def DifferenceOfSquares : Pattern<
  (Poly_SubOp (Poly_MulOp:$lhs $x, $x), (Poly_MulOp:$rhs $y, $y)),
  [
    (Poly_AddOp:$sum $x, $y),
    (Poly_SubOp:$diff $x, $y),
    (Poly_MulOp:$res $sum, $diff),
  ],
  [(HasOneUse:$lhs), (HasOneUse:$rhs)]
>;
```