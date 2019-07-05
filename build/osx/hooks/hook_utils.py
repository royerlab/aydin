import os
import sys
import re
from PyInstaller.utils.hooks import collect_data_files, collect_dynamic_libs, exec_statement

def _module_path(mod):
    return  exec_statement(
        """
        import sys
        import os;
        _tmp = sys.stdout
        sys.stdout = open(os.devnull,"w")
        sys.stderr = open(os.devnull,"w")
        import %s;
        sys.stdout = _tmp
        print(os.path.dirname(%s.__file__))
        """%(mod, mod))

def _get_toc_objects(root ,
                     filter_str = ".*",
                     dir_prefix = "",
                     flatten_dir = False,
                     ):
    reg = re.compile(filter_str)
    res = []
    for fold, subs, files in os.walk(root):

        rel_dir = os.path.relpath(fold,root)
        for fName in files:
            if reg.match(fName):
                if not flatten_dir:
                    name = os.path.join(dir_prefix,rel_dir, fName)
                else:
                    name = os.path.join(dir_prefix, fName)
                res += [(os.path.join(fold,fName), name)]
    return res


def _my_collect_data_files(modname, flatten_dirs = False, **kwargs):
    files = collect_data_files(modname, **kwargs)
    if flatten_dirs:
        # files = [(source, os.path.split(dest)[0])for source, dest in files]
        files = [(source, ".") for source, dest in files]

    return files


if __name__ == '__main__':

    print(_module_path("pyopencl"))
    print(_module_path("PyQt5"))
    # print(_module_path("lightgbm"))


    # print _get_toc_objects(os.path.join(_module_path("pyopencl"), "cl"),
    #                    dir_prefix = "pyopencl/cl")

    print(collect_dynamic_libs("PyQt5"))
    print(collect_dynamic_libs("pyopencl"))
    # print(collect_dynamic_libs("lightgbm"))

    # print _my_collect_data_files("pyopencl", include_py_files = True)

