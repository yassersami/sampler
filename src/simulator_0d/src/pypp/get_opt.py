import getopt
import os


def for_recdir(argv):
    '''
    Get sys.argv options:

    1.1 -l --local: no args
        Do main in local path

    1.2 -d --dir: needs path
        Do main at given -d value

    2. -r --recursive: no args
        Do main at given -d value recursively
    '''

    local_on = False
    dir_on = False
    recursive_on = False

    shortopts = "hld:r"  # : follows opts requiring values
    longopts = ["help", "local", "dir=", "recursive"]  # = same as :
    help_msg = (
        'Posssible options are:\n'
        + f'shortopts: {shortopts}\n'
        + f'longopts: {longopts}\n'
    )

    try:
        opts, args = getopt.getopt(
            args=argv[1:],
            shortopts=shortopts,
            longopts=longopts
        )
    except getopt.GetoptError as err:
        raise AttributeError(
            err
            + '\n' + help_msg
        )

    if not opts:
        raise AttributeError(
            'Can not run code without options.'
            + '\n' + help_msg
        )

    for opt, arg in opts:
        if opt in ["-h", "--help"]:
            print(help_msg)
            exit()
        elif opt in ["-l", "--local"]:
            local_on = True
            simu_dir = None
        elif opt in ["-d", "--dir"]:
            dir_on = True
            simu_dir = os.path.abspath(arg)
        if opt in ["-r", "--recursive"]:
            recursive_on = True

    if local_on and dir_on:
        raise AttributeError(
            'Can not use default directory and given directory.'
            + '\n' + help_msg
        )

    if not dir_on and recursive_on:
        raise AttributeError(
            'Can not use recursive mode without directory.'
            + '\n' + help_msg
        )

    # print(f'opts: {opts}')
    # print(f'args: {args}')

    return dict(
        local_on=local_on,
        dir_on=dir_on,
        simu_dir=simu_dir,
        recursive_on=recursive_on
    )


def for_dir(argv):
    '''
    Get sys.argv options with 2 possible cases:

    1.1 -l --local: no args
        Do main in local path

    1.2 -d --dir: needs path
        Do main at given -d value
    '''

    shortopts = "hld:"  # : follows opts requiring values
    longopts = ["help", "local", "dir="]  # = same as :
    help_msg = (
        'Posssible options are:\n'
        + f'shortopts: {shortopts}\n'
        + f'longopts: {longopts}\n'
        + 'Choose either local or given directory:'
        + '-l --local consider default dir.\n'
        + '-d --dir is map_dir directory path. Takes one arg.\n'
        + 'Syntax:\n'
        + f'python {os.path.abspath(argv[0])} -d root_dir/basename \n'
    )

    try:
        opts, args = getopt.getopt(
            args=argv[1:],
            shortopts=shortopts,
            longopts=longopts
        )
    except getopt.GetoptError as err:
        raise AttributeError(
            err
            + '\n' + help_msg
        )

    if not opts:
        raise AttributeError(
            'Can not run code without options.'
            + '\n' + help_msg
        )

    local_on = False
    dir_on = False

    for opt, arg in opts:
        if opt in ["-h", "--help"]:
            print(help_msg)
            exit()
        elif opt in ["-l", "--local"]:
            local_on = True
            dir = None
        elif opt in ["-d", "--dir"]:
            dir_on = True
            dir = os.path.abspath(arg)

    if local_on and dir_on:
        raise AttributeError(
            'Can not use default and given directory.'
            + '\n' + help_msg
        )

    # print(f'opts: {opts}')
    # print(f'args: {args}')

    return dict(
        local_on=local_on,
        dir_on=dir_on,
        dir=dir
    )


def for_prefix(argv):
    '''
    Get sys.argv options:

    1.1 -l --local: no args
        Do main in local path

    1.2 -d --dir: needs path
        Do main at given -d value

    2. -p --prefix: needs prefix
        Extract from simu folder where dir-name starts with prefix
    '''

    shortopts = "hld:p:"  # : follows opts requiring values
    longopts = ["help", "local", "dir=", "prefix="]  # = same as :
    help_msg = (
        'Posssible options are:\n'
        + f'shortopts: {shortopts}\n'
        + f'longopts: {longopts}\n'
        + 'Choose either local or given directory:'
        + '-l --local consider default dir.\n'
        + '-d --dir is map_dir directory path. Takes one arg.\n'
        + '-p --pref is prefix of simu_dir. Takes one arg.\n'
        + 'Syntax:\n'
        + f'python {os.path.abspath(argv[0])} -d root_dir/basename -p simu_\n'
    )

    try:
        opts, args = getopt.getopt(
            args=argv[1:],
            shortopts=shortopts,
            longopts=longopts
        )
    except getopt.GetoptError as err:
        raise AttributeError(
            err
            + '\n' + help_msg
        )

    if not opts:
        raise AttributeError(
            'Can not run code without options.'
            + '\n' + help_msg
        )

    local_on = False
    dir_on = False

    for opt, arg in opts:
        if opt in ["-h", "--help"]:
            print(help_msg)
            exit()
        elif opt in ["-l", "--local"]:
            local_on = True
            dir = None
        elif opt in ["-d", "--dir"]:
            dir_on = True
            dir = os.path.abspath(arg)
        elif opt in ["-p", "--prefix"]:
            prefix = arg

    if local_on and dir_on:
        raise AttributeError(
            'Can not use default and given directory.'
            + '\n' + help_msg
        )

    # print(f'opts: {opts}')
    # print(f'args: {args}')

    return dict(
        local_on=local_on,
        dir_on=dir_on,
        dir=dir,
        prefix=prefix,
    )


def for_merge(argv):
    '''
    Get sys.argv options with 2 possible cases:

    1. -d --dst: needs one path
        Path to destination folder

    2. -s --src: needs multiple paths
        Paths of source folders to merge
    '''

    shortopts = "hd:s"  # : follows opts requiring values
    longopts = ["help", "dst=", "src"]  # = same as :
    help_msg = (
        'Posssible options are:\n'
        + f'shortopts: {shortopts}\n'
        + f'longopts: {longopts}\n'
        + '-d --dst is destination path. Takes only one arg.\n'
        + '-s --src are source paths. Takes multiple args.\n'
        + 'Syntax:\n'
        + f'python {os.path.abspath(argv[0])} '
        + '-d path/merge_1_2 -s path/run_1 path/run_2\n'
    )

    try:
        opts, args = getopt.getopt(
            args=argv[1:],
            shortopts=shortopts,
            longopts=longopts
        )
    except getopt.GetoptError as err:
        raise AttributeError(
            err
            + '\n' + help_msg
        )

    if not opts:
        raise AttributeError(
            'Can not run code without options.'
            + '\n' + help_msg
        )

    # print(f'opts: {opts}')
    # print(f'args: {args}')

    is_src = False
    src = []
    for opt, arg in opts:
        if opt in ["-h", "--help"]:
            print(help_msg)
            exit()
        elif opt in ["-d", "--dst"]:
            dst = os.path.abspath(arg)
            is_src = False
        elif opt in ["-s", "--src"]:
            is_src = True

    if is_src:
        src = list(map(os.path.abspath, args))
    else:
        raise AttributeError(
            'Last option must be -s.'
            + '\n' + help_msg
        )
    for path in src:
        if not os.path.isdir(path):
            raise ValueError(
                f'Directory "{path}" is not found.'
                + '\n' + help_msg
            )

    return dict(
        dst=dst,
        src=src,
    )


def for_plot(argv):
    '''
    Get sys.argv options with 2 possible cases:

    1. -s --src:
        Path of source folders to merge

    2. -d --dst:
        Path of destination folder
    '''

    shortopts = "hs:d:"
    longopts = ["help", "src=", "dst="]  # it needs local_on and dir...
    help_msg = (
        'Posssible options are:\n'
        + f'shortopts: {shortopts}\n'
        + f'longopts: {longopts}\n'
        + '-s --src is source path.\n'
        + '-d --dst is destination path.\n'
        + 'Syntax:\n'
        + f'python {os.path.abspath(argv[0])} '
        + '-s path/simu_007/ -d path/dash_plot/ \n'
    )

    try:
        opts, args = getopt.getopt(
            args=argv[1:],
            shortopts=shortopts,
            longopts=longopts
        )
    except getopt.GetoptError as err:
        raise AttributeError(
            err
            + '\n' + help_msg
        )

    if not opts:
        raise AttributeError(
            'Can not run code without options.'
            + '\n' + help_msg
        )

    for opt, arg in opts:
        if opt in ["-h", "--help"]:
            print(help_msg)
            exit()
        elif opt in ["-s", "--src"]:
            src = os.path.abspath(arg)
        elif opt in ["-d", "--dst"]:
            dst = os.path.abspath(arg)

    for path in [src, dst]:
        if not os.path.isdir(path):
            raise ValueError(
                f'Directory "{path}" is not found.'
                + '\n' + help_msg
            )

    return dict(
        src=src,
        dst=dst,
    )


def input_dir(local_on, dir):
    '''
    Set simulation inputs' directory
    
    Arguments
    ---------
    local_on: bool
        True to set local simu_dir path.
    dir: string
        path to simu_dir. Necessary if local_on is False.

    Return
    ------
    PATH_simu: string
        Path of simulation directory
    '''
    if local_on:
        PATH_simu = os.path.abspath(os.path.join(
            os.path.abspath(__file__),
            '..', '..', '..',
            'data', 'inputs_folder'
        ))
    else:
        PATH_simu = dir
    
    return PATH_simu


def output_dir(simu_dir):
    return os.path.abspath(os.path.join(simu_dir, 'outputs'))
