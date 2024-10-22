---
title: TVM Python/C++ Interaction
date: 2024/10/16 10:00:12
categories: TVM
tags: TVM learning
excerpt: Code reading of TVM python-C++ interaction.
mathjax: true
katex: true
---

# PackedFunc

TVM 即可以在 C++ 端定义函数然后从 Python 端调用，反之亦可以。这一切都与 `tvm/include/tvm/runtime/packed_func.h` 中的 [PackedFunc]("https://github.com/apache/tvm/blob/main/include/tvm/runtime/packed_func.h#L141") 类有关。

## Constructor 

我们先来看他的构造函数，用于将一个能转换成形如 `std::function<void(TVMArgs, TVMRetValue*)>` 的 `TCallable` 类型转包装成 PackedFunc，返回一个 `PackedFuncSubObj` 对象的智能指针。

并且重载了 `()` 运算符，可以像调用函数一样调用 PackedFunc 对象。具体解释见下面注释。

```c++
template <typename TCallable,
        typename = std::enable_if_t<
            std::is_convertible<TCallable, std::function<void(TVMArgs, TVMRetValue*)>>::value &&  // // 检查 TCallable 是否可以转换为能够接受 TVMArgs 和 TVMRetValue* 两个参数 的函数
            !std::is_base_of<TCallable, PackedFunc>::value>>
explicit PackedFunc(TCallable data) {
using ObjType = PackedFuncSubObj<TCallable>;
data_ = make_object<ObjType>(std::forward<TCallable>(data));  // 使用完美转发创建对象
}

template <typename T, typename... Args>  // ... 包展开运算符，可以接受0个或多个参数
inline ObjectPtr<T> make_object(Args&&... args) {
  return SimpleObjAllocator().make_object<T>(std::forward<Args>(args)...); // 创建并返回一个 PackedFuncSubObj 对象的智能指针

template <typename... Args>
  inline TVMRetValue operator()(Args&&... args) const;  // 使得对象可以像函数一样被调用

  TVM_ALWAYS_INLINE void CallPacked(TVMArgs args, TVMRetValue* rv) const;
  /*! \return Whether the packed function is nullptr */
  bool operator==(std::nullptr_t null) const { return data_ == nullptr; }
  /*! \return Whether the packed function is not nullptr */
  bool operator!=(std::nullptr_t null) const { return data_ != nullptr; }

  TVM_DEFINE_OBJECT_REF_METHODS(PackedFunc, ObjectRef, PackedFuncObj);
}
```

`()` 操作符重载的实现如下

```c++
/* Implementation of () Operator Reloaded */
template <typename... Args>
inline TVMRetValue PackedFunc::operator()(Args&&... args) const {
    const int kNumArgs = sizeof...(Args);//sizeof...(Args)表示获取可变参数数量。
    const int kArraySize = kNumArgs > 0 ? kNumArgs : 1;
    TVMValue values[kArraySize];
    int type_codes[kArraySize];
    // 展开可变参数并使用 TVMArgsSetter 赋值
    // TVMArgsSetter 函数的作用是将调用 PackedFunc 传入的参数转化为 TVMValue 类型。
    detail::for_each(TVMArgsSetter(values, type_codes), std::forward<Args>(args)...);
    TVMRetValue rv;
    //获取指针并转换为 PackedFuncObj 对象
    //接着构造 TVMArgs 类。将不同输入参数转化为统一的类型无关调用格式
    //传递给CallPacked完成PackedFunc调用。
    (static_cast<PackedFuncObj*>(data_.get()))->CallPacked(TVMArgs(values, type_codes, kNumArgs), &rv);
    return rv;
}
```

## TVMArgs & TVMRetValue

下面我们来看 [TVMArgs]("https://github.com/apache/tvm/blob/main/include/tvm/runtime/packed_func.h#L405") 和 [TVMRetValue]("https://github.com/apache/tvm/blob/main/include/tvm/runtime/packed_func.h#L946") 的定义。

`TVMArgs` 构造函数接受三个参数，分别代表参数值，参数值对应的数据类型代码，参数个数。
`TVMValue` 是一个 union，使用 `DLDataType` 来描述 TVMArgs 中数据的类型 (类型代码，位数，向量长度)，使用 `DLDevice` 来描述该数据被存在哪种硬件上。他俩都位于第三方 dlpack 包中。
```c++
class TVMArgs {
 public:
    const TVMValue* values;
    const int* type_codes;
    int num_args;

    TVMArgs(const TVMValue* values, const int* type_codes, int num_args)
        : values(values), type_codes(type_codes), num_args(num_args) {}
    inline int size() const;  // 返回参数个数
    inline TVMArgValue operator[](int i) const;  // 重载 [] 操作符，返回第 i 个参数的值

    template <typename T>
    inline T At(int i) const;
};

typedef union {
  int64_t v_int64;
  double v_float64;
  void* v_handle;
  const char* v_str;
  DLDataType v_type;  // dlpack: {uint8_t code; uint8_t bits; uint16_t lanes;}
  DLDevice v_device;  // dlpack: {DLDeviceType device_type; int32_t device_id}
} TVMValue;
```

`TVMRetValue` 继承自 `TVMPODValue_CRTP_<TVMRetValue>`，基类为 `TVMPODValue_`，从名字看出他与 C++ 中的 Plain Old Data 行为类似。

{% fold info@Curious Recurring Template Pattern (CRTP) %}
Curiously Recurring Template Pattern (CRTP，奇异递归模板模式) 是一种 C++ 模板元编程技巧，它允许在基类中使用派生类的类型信息，从而实现静态多态性，避免运行时类型检查的开销。

CRTP 的核心在于，派生类将自身作为模板参数传递给基类。 这使得基类可以在编译时访问派生类的类型信息，从而在基类中实现一些依赖于派生类类型的方法。

```c++
template <typename Derived>
class Base {
public:
  void print() {
    static_cast<Derived*>(this)->print_impl(); // 调用派生类的方法
  }
};

class Derived : public Base<Derived> {
public:
  void print_impl() {
    std::cout << "Hello from Derived!" << std::endl;
  }
};

int main() {
  Derived d;
  d.print(); // 输出 "Hello from Derived!"
  return 0;
}
```

- Base` 是一个模板类，它接受一个类型参数 `Derived`.
- `Derived` 继承自 `Base<Derived>`，将自身作为模板参数传递给 Base。
- `Base` 类中的 `print()` 方法使用 `static_cast` 将 this 指针转换为 `Derived` 类型指针，然后调用 `print_impl()` 方法。 `print_impl()` 方法是在 `Derived `类中定义的。
{% endfold %}


1. 重载了各种类型转换运算符 (例如 `operator void*()`, `operator DLTensor*()`, `operator NDArray()`...)，允许以类型安全的方式访问 `TVMValue` 中存储的值并转换为对应的类型。
2. `TryAsBool()`, `TryAsInt()`, `TryAsFloat()` 等辅助函数提供了一种更安全的方式来尝试将 `TVMValue` 转换为布尔值、整数或浮点数，并返回 `std::optional` 来指示转换是否成功。

```c++
class TVMPODValue_ {
 public:
    /* Public Function */

 protected:
    friend class TVMArgsSetter;
    friend class TVMRetValue;
    friend class TVMMovableArgValue_;
    TVMPODValue_() : type_code_(kTVMNullptr) {}
    TVMPODValue_(TVMValue value, int type_code) : value_(value), type_code_(type_code) {}

    /*! \brief The value */
    TVMValue value_;
    /*! \brief the type code */
    int type_code_;
};
```

# C++ Function Register

C++ 端的函数注册宏定义是 [TVM_REGISTER_GLOBAL](https://github.com/apache/tvm/blob/main/include/tvm/runtime/registry.h#L384)，将生成的唯一变量名赋值为 `::tvm::runtime::Registry::Register(OpName)` 的返回值。

```c++
#define TVM_REGISTER_GLOBAL(OpName) \
  TVM_STR_CONCAT(TVM_FUNC_REG_VAR_DEF, __COUNTER__) = ::tvm::runtime::Registry::Register(OpName)
```

Register 函数会返回一个 [Registry](https://github.com/apache/tvm/blob/main/include/tvm/runtime/registry.h#L146) 类，提供多种方法来设置函数体。


```c++
class Registry {
 public:
  // 接受一个 PackedFunc 对象作为参数
  Registry& set_body(PackedFunc f);  // 以及一个重载
  // 专门用于设置类型安全的函数体, 使用 TypedPackedFunc
  template <typename FLambda> Registry& set_body_typed(FLambda f);
  // 注册类 T 的 一个参数类型为 Args...,返回值为 R 的成员函数
  template <typename T, typename R, typename... Args> Registry& set_body_method(R (T::*f)(Args...));
​  
  //使用 name 完成调用注册函数实现函数的注册
  static Registry& Register(const std::string& name);
  //在哈希表中寻找名字为name的函数并返回
  static const PackedFunc* Get(const std::string& name);
  //创建函数名列表
  static std::vector ListNames();
​
 protected:
  std::string name_;
  PackedFunc func_;
  friend struct Manager;
};
```

[Manager](https://github.com/apache/tvm/blob/main/src/runtime/registry.cc#L39) 的定义如下。
- `fmap` 用于存储已注册的函数
- `mutex`是一个互斥锁，用于保护 fmap 的线程安全。
- `Global()` 静态方法，返回一个全局的静态 Manager 对象 (只在第一次调用时被初始化).

```c++
struct Registry::Manager {

  std::unordered_map<String, Registry*> fmap;
  // mutex
  std::mutex mutex;

  Manager() {}

  static Manager* Global() {
    static Manager* inst = new Manager();
    return inst;
  }
};
```

`Registry::Register` 函数定义如下。使用 Manager 类来管理已注册的函数，并使用互斥锁来保证线程安全。 它允许注册新的函数，并检查函数名是否冲突，以及是否允许覆盖已存在的函数。 返回 Registry 对象的引用方便后续设置函数体等操作。

```c++
Registry& Registry::Register(const String& name, bool can_override) { // NOLINT()
    Manager m = Manager::Global();
    std::lock_guardstd::mutex lock(m->mutex);
    if (m->fmap.count(name)) {
        ICHECK(can_override) << "Global PackedFunc " << name << " is already registered";
    }

    Registry* r = new Registry();
    r->name_ = name;
    m->fmap[name] = r;
    return *r;
}
```

{% fold info@An Register Example in C++%}
```c++
struct Example {
    int doThing(int x);
}

TVM_REGISTER_GLOBAL("Example_doThing")
    .set_body_method(&Example::doThing); // will have type int(Example, int)
```
{% endfold %}

# Python Call C++ Function

python 端调用 C++ 函数最终都会进入到 [_init_api_prefix](https://github.com/apache/tvm/blob/main/include/tvm/runtime/packed_func.h#L405") 函数


```python
def _init_api_prefix(module_name, prefix):
    module = sys.modules[module_name]

    for name in list_global_func_names():
        if not name.startswith(prefix):
            continue

        fname = name[len(prefix) + 1 :]
        target_module = module

        if fname.find(".") != -1:
            continue
        f = get_global_func(name)  # 会调用 _get_global_func
        ff = _get_api(f)
        ff.__name__ = fname
        ff.__doc__ = "TVM PackedFunc %s. " % fname
        setattr(target_module, ff.__name__, ff)
```

`get_global_func` 会调用 [_get_global_func](https://github.com/apache/tvm/blob/58a43c87245e58ee09f2cdbde26fb2cc5167df9d/python/tvm/_ffi/_ctypes/packed_func.py#L292) 返回一个 python 端的 `PackedFunc` 对象。

```python
def _get_global_func(name, allow_missing=False):
    handle = PackedFuncHandle()
    check_call(_LIB.TVMFuncGetGlobal(c_str(name), ctypes.byref(handle)))

    if handle.value:
        return _make_packed_func(handle, False)  # 返回一个 python 中的 PackedFunc 对象

    if allow_missing:
        return None

    raise ValueError("Cannot find global function %s" % name)

def _make_packed_func(handle, is_global):
    """Make a packed function class"""
    obj = _CLASS_PACKED_FUNC.__new__(_CLASS_PACKED_FUNC)
    obj.is_global = is_global
    obj.handle = handle
    return obj
```

上面代码中 _CLASS_PACKED_FUNC 是一个全局变量， 在 `/python/tvm/runtime/packed_func.py` 中会调用 [_set_class_packed_func](https://github.com/apache/tvm/blob/58a43c87245e58ee09f2cdbde26fb2cc5167df9d/python/tvm/runtime/packed_func.py#L65) 将其设置为 python 端的 `PackedFunc` 类。

python 端的 `PackedFunc` 类继承自 [PackedFuncBase](https://github.com/apache/tvm/blob/main/python/tvm/_ffi/_ctypes/packed_func.py#L199) 类，它本质上就是在 python 端定义了一个和 C++ 端具有相同形式的类，将传入的参数转换成 C++ 端的 TVMArgs, 再通过 ctype 加载的 lib 调用 C++ 端的 PackedFunc 对象，并将返回值转换成 python 端的 TVMRetValue. 

```python
class PackedFuncBase(object):
    """Function base."""

    __slots__ = ["handle", "is_global"]

    # pylint: disable=no-member
    def __init__(self, handle, is_global):
        self.handle = handle
        self.is_global = is_global

    def __del__(self):
        if not self.is_global and _LIB is not None:
            if _LIB.TVMFuncFree(self.handle) != 0:
                raise_last_ffi_error()

    def __call__(self, *args):
        temp_args = []
        values, tcodes, num_args = _make_tvm_args(args, temp_args)  # convert to TVMArgs
        ret_val = TVMValue()
        ret_tcode = ctypes.c_int()
        if (
            _LIB.TVMFuncCall(  # Call C++ function
                self.handle,
                values,
                tcodes,
                ctypes.c_int(num_args),
                ctypes.byref(ret_val),
                ctypes.byref(ret_tcode),
            )
            != 0
        ):
            raise_last_ffi_error()
        _ = temp_args
        _ = args
        return RETURN_SWITCH[ret_tcode.value](ret_val)
```

# C++ Call Python Function
[register_func](https://github.com/apache/tvm/blob/58a43c87245e58ee09f2cdbde26fb2cc5167df9d/python/tvm/_ffi/registry.py#L158) 实现注册 python 端的函数。如果 f 已经提供，则直接注册并返回已注册的函数。如果 f 未提供，则返回一个 register 函数以延迟注册 (通常以 decorator 形式存在).

```python
def register_func(func_name, f=None, override=False):
    if callable(func_name):
        f = func_name
        func_name = f.__name__

    if not isinstance(func_name, str):
        raise ValueError("expect string function name")

    ioverride = ctypes.c_int(override)

    def register(myf):
        """internal register function"""
        if not isinstance(myf, PackedFuncBase):
            myf = convert_to_tvm_func(myf)  # Python -> C++ PackedFunc
        check_call(_LIB.TVMFuncRegisterGlobal(c_str(func_name), myf.handle, ioverride))
        return myf

    if f:
        return register(f)
    return register
```

内部的 register 函数通过 [convert_to_tvm_func](https://github.com/apache/tvm/blob/main/python/tvm/_ffi/_ctypes/packed_func.py#L61) 将一个 Python 函数转换为 TVM 函数 (tvm.nd.Function)，与 C++ 代码进行交互。

内部定义的 cfun 是一个 C++ 风格的回调函数，它将接收的 C++ 参数转换为 Python 参数，调用 pyfunc，并将结果返回给 C++ 端。

```python
def convert_to_tvm_func(pyfunc):
    """Convert a python function to TVM function

    Parameters
    ----------
    pyfunc : python function
        The python function to be converted.

    Returns
    -------
    tvmfunc: tvm.nd.Function
        The converted tvm function.
    """
    local_pyfunc = pyfunc

    def cfun(args, type_codes, num_args, ret, _):
        """ctypes function"""
        num_args = num_args.value if isinstance(num_args, ctypes.c_int) else num_args
        pyargs = (C_TO_PY_ARG_SWITCH[type_codes[i]](args[i]) for i in range(num_args))
        # pylint: disable=broad-except
        try:
            rv = local_pyfunc(*pyargs)
        except Exception as err:
            msg = traceback.format_exc()
            msg = py2cerror(msg)
            _LIB.TVMAPISetLastPythonError(ctypes.py_object(err))

            return -1

        if rv is not None:
            if isinstance(rv, tuple):
                raise ValueError("PackedFunction can only support one return value")
            temp_args = []
            values, tcodes, _ = _make_tvm_args((rv,), temp_args)
            if not isinstance(ret, TVMRetValueHandle):
                ret = TVMRetValueHandle(ret)
            if _LIB.TVMCFuncSetReturn(ret, values, tcodes, ctypes.c_int(1)) != 0:
                raise_last_ffi_error()
            _ = temp_args
            _ = rv
        return 0

    handle = PackedFuncHandle()
    f = TVMPackedCFunc(cfun)  #  转换为一个 C 类型的回调函数
    pyobj = ctypes.py_object(f)
    ctypes.pythonapi.Py_IncRef(pyobj)
    if _LIB.TVMFuncCreateFromCFunc(f, pyobj, TVM_FREE_PYOBJ, ctypes.byref(handle)) != 0:
        raise_last_ffi_error()
    return _make_packed_func(handle, False)
```

注册完成之后该函数已经存在于 Manager 中，可以在 C++ 端调用如下调用

```c++
tvm::runtime::PackedFunc f = tvm::runtime::Registry::Get("my_python_function");

/**/
const PackedFunc* Registry::Get(const String& name) {
  Manager* m = Manager::Global();
  std::lock_guard<std::mutex> lock(m->mutex);
  auto it = m->fmap.find(name);
  if (it == m->fmap.end()) return nullptr;
  return &(it->second->func_);
}
```

{% fold info@An Register Example in Python %}

```python
def callback(msg):
  print(msg)

# convert to PackedFunc
tvm.register_func("my_call_back", callback)
```
{% endfold %}
# Summary

TVM 的互调机制可以简述为：在 C++ 和 Python 两边使用了一个统一的函数原型 `void(TVMArgs args, TVMRetValue *rv)`，这就是`PackedFunc` 类的机制，实现主要是重载了函数调用运算符 `()`，真正的函数体是通过 set_body 去设置的。

相互调用其实是每次去全局注册函数表中寻找相应的函数名，然后做两种语言之间PackedFunc对象的转换，再去执行。