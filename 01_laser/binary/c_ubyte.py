import ctypes

a = ctypes.c_ubyte(1)
a = ctypes.c_ushort(a)
print(a)


b = int('1', 16)
print(b)