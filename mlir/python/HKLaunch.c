/*
 * hk._mlir._mlir_libs._hkLaunch: a code-object kernel's launch in C
 * (hk.runtime.codeobj.CodeObjectKernel.launcher).
 *
 * The Python launcher spends what is left of a launch's host time reading
 * five attributes of each tensor through the interpreter and packing the
 * argument with struct. This does the same reads through the C API -- the
 * same ones include/pyutils/hk_bind.cuh does for a pybind entry point -- and
 * writes the argument straight into a buffer: the pybind entry point's cost,
 * for a kernel that has no host half to build.
 *
 * It takes only the common call: every tensor of the kernel's dtype, on the
 * current GPU, contiguous, 1 to 4 dimensions. Anything else -- including the
 * fast paths being off (hk.runtime.hostpath) -- goes to `slow`, the Python
 * launcher, which checks everything and raises what it always raised.
 *
 * No HIP headers and no HIP library: hipModuleLaunchKernel is called through
 * the address the Python side read from the HIP runtime this process already
 * runs on (codeobj.hip()), so there is never a second copy of the runtime.
 * Plain CPython API, so it builds with the rest of hk._mlir and ships in the
 * wheel.
 */

#define PY_SSIZE_T_CLEAN
#include <Python.h>
#include <stddef.h>
#include <stdint.h>
#include <string.h>

typedef int (*launch_fn_t)(void *f, unsigned gx, unsigned gy, unsigned gz,
                           unsigned bx, unsigned by, unsigned bz, unsigned shmem,
                           void *stream, void **params, void **extra);

/* hipModuleLaunchKernel's `extra` keys (hip_runtime_api.h). */
#define HK_PARAM_BUFFER ((void *)1)
#define HK_PARAM_SIZE ((void *)2)
#define HK_PARAM_END ((void *)3)

#define HK_MAX_TENSORS 32
#define HK_MAX_PROG 64
#define HK_MAX_ARG 4096

enum { OP_CONST = 0, OP_DIM = 1, OP_ADD = 2, OP_SUB = 3, OP_MUL = 4, OP_DIV = 5, OP_MOD = 6 };

typedef struct {
    PyObject_HEAD
    vectorcallfunc vectorcall;
    int n;
    PyObject *dtypes[HK_MAX_TENSORS];
    Py_ssize_t size;
    /* per tensor: pointer, batch, depth, rows, cols, row stride offsets */
    Py_ssize_t off[HK_MAX_TENSORS][6];
    /* each grid dimension: a postfix program over the tensors' padded extents */
    long long prog[3][HK_MAX_PROG];
    int prog_len[3];
    unsigned block, shmem;
    launch_fn_t launch;
    PyObject *get_stream;   /* idx -> hipStream_t as an int (torch's current one) */
    PyObject *get_device;   /* () -> the current device */
    PyObject *function;     /* idx -> the kernel's hipFunction_t as an int */
    PyObject *on_error;     /* (err) -> raises */
    PyObject *slow;         /* the Python launcher: everything else */
    PyObject *hostpath;     /* hk.runtime.hostpath: its ENABLED */
    void *fns[64];          /* hipFunction_t by device index, filled on first use */
} Launcher;

static PyObject *s_dtype, *s_shape, *s_get_device, *s_is_contiguous, *s_data_ptr, *s_enabled;

static void launcher_dealloc(Launcher *self) {
    for (int i = 0; i < self->n; ++i) Py_XDECREF(self->dtypes[i]);
    Py_XDECREF(self->get_stream);
    Py_XDECREF(self->get_device);
    Py_XDECREF(self->function);
    Py_XDECREF(self->on_error);
    Py_XDECREF(self->slow);
    Py_XDECREF(self->hostpath);
    Py_TYPE(self)->tp_free((PyObject *)self);
}

static PyObject *to_slow(Launcher *self, PyObject *const *args, size_t nargsf) {
    return PyObject_Vectorcall(self->slow, args, PyVectorcall_NARGS(nargsf), NULL);
}

static long long call_long(PyObject *obj, PyObject *name, int *ok) {
    PyObject *r = PyObject_CallMethodNoArgs(obj, name);
    if (r == NULL) { *ok = 0; return 0; }
    long long v = PyLong_AsLongLong(r);
    Py_DECREF(r);
    if (v == -1 && PyErr_Occurred()) *ok = 0;
    return v;
}

static int truthy_call(PyObject *obj, PyObject *name) {
    PyObject *r = PyObject_CallMethodNoArgs(obj, name);
    if (r == NULL) return -1;
    int t = PyObject_IsTrue(r);
    Py_DECREF(r);
    return t;
}

/* A grid dimension, as the traced expression computes it (HostExpr.value: Python's integer
 * arithmetic, floor division included). 0 where the program is malformed or divides by zero:
 * the caller then leaves the call to Python, which raises what it raises. */
static int eval_prog(const long long *pr, int len, long long ext[][4], int n, long long *out) {
    long long st[HK_MAX_PROG];
    int sp = 0;
    for (int k = 0; k < len;) {
        long long op = pr[k++];
        if (op == OP_CONST) {
            if (k >= len) return 0;
            st[sp++] = pr[k++];
            continue;
        }
        if (op == OP_DIM) {
            if (k + 1 >= len || pr[k] < 0 || pr[k] >= n || pr[k + 1] < 0 || pr[k + 1] > 3) return 0;
            st[sp++] = ext[pr[k]][pr[k + 1]];
            k += 2;
            continue;
        }
        if (sp < 2) return 0;
        long long b = st[--sp], a = st[--sp], r;
        if ((op == OP_DIV || op == OP_MOD) && b == 0) return 0;
        if (op == OP_ADD) r = a + b;
        else if (op == OP_SUB) r = a - b;
        else if (op == OP_MUL) r = a * b;
        else if (op == OP_DIV) {
            r = a / b;
            if (a % b != 0 && ((a < 0) != (b < 0))) --r;
        } else if (op == OP_MOD) {
            r = a % b;
            if (r != 0 && ((r < 0) != (b < 0))) r += b;
        } else {
            return 0;
        }
        st[sp++] = r;
    }
    if (sp != 1) return 0;
    *out = st[0];
    return 1;
}

static PyObject *launcher_call(PyObject *callable, PyObject *const *args, size_t nargsf,
                               PyObject *kwnames) {
    Launcher *self = (Launcher *)callable;
    Py_ssize_t nargs = PyVectorcall_NARGS(nargsf);
    if (kwnames != NULL || nargs != self->n) return to_slow(self, args, nargsf);

    PyObject *en = PyObject_GetAttr(self->hostpath, s_enabled);
    if (en == NULL) return NULL;
    int on = PyObject_IsTrue(en);
    Py_DECREF(en);
    if (on < 0) return NULL;
    if (!on) return to_slow(self, args, nargsf);

    long long ext[HK_MAX_TENSORS][4];
    unsigned long long ptr[HK_MAX_TENSORS];
    long long idx = -1;
    int ok = 1;
    for (int i = 0; i < self->n; ++i) {
        PyObject *t = args[i];
        PyObject *dt = PyObject_GetAttr(t, s_dtype);
        if (dt == NULL) { PyErr_Clear(); return to_slow(self, args, nargsf); }
        int same = dt == self->dtypes[i];
        Py_DECREF(dt);
        if (!same) return to_slow(self, args, nargsf);
        long long dev = call_long(t, s_get_device, &ok);
        if (!ok) { PyErr_Clear(); return to_slow(self, args, nargsf); }
        if (i == 0) idx = dev;
        if (dev < 0 || dev != idx || dev >= 64) return to_slow(self, args, nargsf);
        int cont = truthy_call(t, s_is_contiguous);
        if (cont != 1) { PyErr_Clear(); return to_slow(self, args, nargsf); }
        PyObject *shape = PyObject_GetAttr(t, s_shape);
        if (shape == NULL || !PyTuple_Check(shape)) {
            Py_XDECREF(shape);
            PyErr_Clear();
            return to_slow(self, args, nargsf);
        }
        Py_ssize_t nd = PyTuple_GET_SIZE(shape);
        if (nd < 1 || nd > 4) { Py_DECREF(shape); return to_slow(self, args, nargsf); }
        ext[i][0] = ext[i][1] = ext[i][2] = 1;
        for (Py_ssize_t d = 0; d < nd; ++d)
            ext[i][4 - nd + d] = PyLong_AsLongLong(PyTuple_GET_ITEM(shape, d));
        Py_DECREF(shape);
        if (PyErr_Occurred()) { PyErr_Clear(); return to_slow(self, args, nargsf); }
        PyObject *p = PyObject_CallMethodNoArgs(t, s_data_ptr);
        if (p == NULL) { PyErr_Clear(); return to_slow(self, args, nargsf); }
        ptr[i] = PyLong_AsUnsignedLongLong(p);
        Py_DECREF(p);
        if (PyErr_Occurred()) { PyErr_Clear(); return to_slow(self, args, nargsf); }
    }

    long long grid[3];
    for (int g = 0; g < 3; ++g)
        if (!eval_prog(self->prog[g], self->prog_len[g], ext, self->n, &grid[g]))
            return to_slow(self, args, nargsf);
    if (grid[0] <= 0 || grid[1] <= 0 || grid[2] <= 0) Py_RETURN_NONE;

    PyObject *cur = PyObject_CallNoArgs(self->get_device);
    if (cur == NULL) return NULL;
    long long current = PyLong_AsLongLong(cur);
    Py_DECREF(cur);
    if (current != idx) { PyErr_Clear(); return to_slow(self, args, nargsf); }

    void *fn = self->fns[idx];
    if (fn == NULL) {
        PyObject *f = PyObject_CallFunction(self->function, "L", idx);
        if (f == NULL) return NULL;
        fn = PyLong_AsVoidPtr(f);
        Py_DECREF(f);
        if (fn == NULL) return PyErr_Occurred() ? NULL : to_slow(self, args, nargsf);
        self->fns[idx] = fn;
    }
    PyObject *s = PyObject_CallFunction(self->get_stream, "L", idx);
    if (s == NULL) return NULL;
    void *stream = PyLong_AsVoidPtr(s);
    Py_DECREF(s);
    if (PyErr_Occurred()) return NULL;

    /* The argument: zeroed (an extent fixed at compile time is an empty member; padding is
       never read), then every field at its offset. hipModuleLaunchKernel copies it out
       before it returns, so the stack holds it. */
    unsigned char buf[HK_MAX_ARG];
    memset(buf, 0, (size_t)self->size);
    for (int i = 0; i < self->n; ++i) {
        memcpy(buf + self->off[i][0], &ptr[i], 8);
        for (int d = 0; d < 4; ++d) {
            int v = (int)ext[i][d];
            memcpy(buf + self->off[i][1 + d], &v, 4);
        }
        int row = (int)ext[i][3];                   /* contiguous: the row stride is the row */
        memcpy(buf + self->off[i][5], &row, 4);
    }
    size_t size = (size_t)self->size;
    void *extra[5] = {HK_PARAM_BUFFER, buf, HK_PARAM_SIZE, &size, HK_PARAM_END};
    int err;
    Py_BEGIN_ALLOW_THREADS
    err = self->launch(fn, (unsigned)grid[0], (unsigned)grid[1], (unsigned)grid[2],
                       self->block, 1, 1, self->shmem, stream, NULL, extra);
    Py_END_ALLOW_THREADS
    if (err) {
        PyObject *r = PyObject_CallFunction(self->on_error, "i", err);
        Py_XDECREF(r);
        return r == NULL ? NULL : Py_NewRef(Py_None);
    }
    Py_RETURN_NONE;
}

static PyTypeObject LauncherType;

/* Launcher(dtypes, size, offsets, programs, block, shmem, launch_address, get_stream,
 *          get_device, function, on_error, slow, hostpath) */
static PyObject *make(PyObject *mod, PyObject *a) {
    PyObject *dtypes, *offsets, *progs, *get_stream, *get_device, *function, *on_error, *slow,
        *hostpath;
    Py_ssize_t size;
    unsigned block, shmem;
    unsigned long long addr;
    if (!PyArg_ParseTuple(a, "OnOOIIKOOOOOO", &dtypes, &size, &offsets, &progs, &block, &shmem,
                          &addr, &get_stream, &get_device, &function, &on_error, &slow, &hostpath))
        return NULL;
    if (!PyTuple_Check(dtypes) || !PyTuple_Check(offsets) || !PyTuple_Check(progs)
        || PyTuple_GET_SIZE(progs) != 3 || PyTuple_GET_SIZE(dtypes) != PyTuple_GET_SIZE(offsets)
        || PyTuple_GET_SIZE(dtypes) > HK_MAX_TENSORS || size <= 0 || size > HK_MAX_ARG) {
        PyErr_SetString(PyExc_ValueError, "_hkLaunch.Launcher: bad layout");
        return NULL;
    }
    Launcher *self = PyObject_New(Launcher, &LauncherType);
    if (self == NULL) return NULL;
    self->vectorcall = launcher_call;
    self->n = (int)PyTuple_GET_SIZE(dtypes);
    self->size = size;
    self->block = block;
    self->shmem = shmem;
    self->launch = (launch_fn_t)(uintptr_t)addr;
    memset(self->fns, 0, sizeof(self->fns));
    for (int i = 0; i < self->n; ++i) {
        self->dtypes[i] = Py_NewRef(PyTuple_GET_ITEM(dtypes, i));
        PyObject *o = PyTuple_GET_ITEM(offsets, i);
        for (int k = 0; k < 6; ++k) {
            self->off[i][k] = PyLong_AsSsize_t(PyTuple_GetItem(o, k));
            Py_ssize_t width = k == 0 ? 8 : 4;           /* the pointer, then five ints */
            if (!PyErr_Occurred() && (self->off[i][k] < 0 || self->off[i][k] + width > size))
                PyErr_SetString(PyExc_ValueError, "_hkLaunch.Launcher: a field outside the argument");
        }
    }
    for (int g = 0; g < 3; ++g) {
        PyObject *p = PyTuple_GET_ITEM(progs, g);
        Py_ssize_t len = PyTuple_Check(p) ? PyTuple_GET_SIZE(p) : -1;
        if (len < 1 || len > HK_MAX_PROG) {
            PyErr_SetString(PyExc_ValueError, "_hkLaunch.Launcher: bad grid program");
            break;
        }
        self->prog_len[g] = (int)len;
        for (Py_ssize_t k = 0; k < len; ++k)
            self->prog[g][k] = PyLong_AsLongLong(PyTuple_GET_ITEM(p, k));
    }
    self->get_stream = Py_NewRef(get_stream);
    self->get_device = Py_NewRef(get_device);
    self->function = Py_NewRef(function);
    self->on_error = Py_NewRef(on_error);
    self->slow = Py_NewRef(slow);
    self->hostpath = Py_NewRef(hostpath);
    if (PyErr_Occurred()) {
        Py_DECREF(self);
        return NULL;
    }
    return (PyObject *)self;
}

static PyTypeObject LauncherType = {
    PyVarObject_HEAD_INIT(NULL, 0)
    .tp_name = "_hkLaunch.Launcher",
    .tp_basicsize = sizeof(Launcher),
    .tp_dealloc = (destructor)launcher_dealloc,
    .tp_flags = Py_TPFLAGS_DEFAULT | Py_TPFLAGS_HAVE_VECTORCALL,
    .tp_vectorcall_offset = offsetof(Launcher, vectorcall),
    .tp_call = PyVectorcall_Call,
    .tp_doc = "A code-object kernel's launch (hk.runtime.codeobj).",
};

/* grid(program, extents): one grid dimension for per-tensor (batch, depth, rows, cols) --
 * what a launch computes, for tests to hold against HostExpr.value. None where the launch
 * would leave the call to Python. */
static PyObject *grid(PyObject *mod, PyObject *a) {
    PyObject *prog, *extents;
    if (!PyArg_ParseTuple(a, "OO", &prog, &extents)) return NULL;
    if (!PyTuple_Check(prog) || !PyTuple_Check(extents) || PyTuple_GET_SIZE(prog) > HK_MAX_PROG
        || PyTuple_GET_SIZE(extents) > HK_MAX_TENSORS) {
        PyErr_SetString(PyExc_ValueError, "_hkLaunch.grid: (program tuple, extents tuple)");
        return NULL;
    }
    long long pr[HK_MAX_PROG], ext[HK_MAX_TENSORS][4];
    int len = (int)PyTuple_GET_SIZE(prog), n = (int)PyTuple_GET_SIZE(extents);
    for (int k = 0; k < len; ++k) pr[k] = PyLong_AsLongLong(PyTuple_GET_ITEM(prog, k));
    for (int i = 0; i < n; ++i)
        for (int d = 0; d < 4; ++d)
            ext[i][d] = PyLong_AsLongLong(PyTuple_GetItem(PyTuple_GET_ITEM(extents, i), d));
    if (PyErr_Occurred()) return NULL;
    long long out;
    if (!eval_prog(pr, len, ext, n, &out)) Py_RETURN_NONE;
    return PyLong_FromLongLong(out);
}

static PyMethodDef methods[] = {
    {"grid", grid, METH_VARARGS, "grid(program, extents): one grid dimension, or None"},
    {"Launcher", make, METH_VARARGS, "Launcher(dtypes, size, offsets, programs, block, shmem, "
     "launch_address, get_stream, get_device, function, on_error, slow, hostpath)"},
    {NULL, NULL, 0, NULL},
};

static struct PyModuleDef module = {PyModuleDef_HEAD_INIT, "_hkLaunch", NULL, -1, methods};

PyMODINIT_FUNC PyInit__hkLaunch(void) {
    if (PyType_Ready(&LauncherType) < 0) return NULL;
    s_dtype = PyUnicode_InternFromString("dtype");
    s_shape = PyUnicode_InternFromString("shape");
    s_get_device = PyUnicode_InternFromString("get_device");
    s_is_contiguous = PyUnicode_InternFromString("is_contiguous");
    s_data_ptr = PyUnicode_InternFromString("data_ptr");
    s_enabled = PyUnicode_InternFromString("ENABLED");
    return PyModule_Create(&module);
}
