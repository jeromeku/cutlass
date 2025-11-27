import os
import logging

PREFIX = "CUTE_DSL"

def set_logging(log_to_console: bool = False, log_level: int = 10, log_to_file: str = None):
    if log_to_console:
        os.environ[f"{PREFIX}_LOG_TO_CONSOLE"] = "1"
    if log_to_file is not None:
        os.environ[f"{PREFIX}_LOG_TO_FILE"] = log_to_file
    os.environ[f"{PREFIX}_LOG_LEVEL"] = str(log_level)


def set_compilation_opts(
    dry_run: bool = False,
    dump_ptx: bool = False,
    ptx_dir: str = f"{PREFIX}_ptx",
    print_ir: bool = False,
    keep_ir: bool = False,
):
    if print_ir:
        os.environ[f"{PREFIX}_PRINT_IR"] = "1"
    if keep_ir:
        os.environ[f"{PREFIX}_KEEP_IR"] = "1"
    if dump_ptx:
        os.environ[f"{PREFIX}_PTX"] = "1"
        os.environ[f"{PREFIX}_DUMP_PTX"] = "1"
        os.environ[f"{PREFIX}_PTX_DIR"] = ptx_dir
    if dry_run:
        os.environ[f"{PREFIX}_DRYRUN"] = "1"


def set_arch(arch: str):
    from cutlass.base_dsl import detect_gpu_arch

    os.environ.setdefault(f"{PREFIX}_ARCH", detect_gpu_arch(None))


def patch_cutlass_env(
    log_to_console: bool = False,
    log_to_file: bool = False,
    logdir: str = "cute_logs",
    log_level: int = logging.DEBUG,
    arch: str = None,
):
    # Need to set arch env flag before importing cutlass
    if arch is not None:
        os.environ[f"{PREFIX}_ARCH"] = arch
    
    import cutlass.base_dsl.utils.logger as cutlass_logger
    from cutlass.base_dsl import BaseDSL
    from cutlass.base_dsl.arch import Arch
    from cutlass.cutlass_dsl import CuTeDSL, T

    arch = arch or CuTeDSL._get_dsl().get_arch_enum()
    
    print(f"Setting arch to {arch}")
    
    log_to_file = log_to_file or os.environ.get(f"{PREFIX}_LOG_TO_FILE", "0") == "1"
    log_to_console = log_to_console or os.environ.get(f"{PREFIX}_LOG_TO_CONSOLE", "0") == "1"

    if not (log_to_console or log_to_file):
        return

    fmt = (
        "%(asctime)s - %(name)s - %(levelname)s - "
        "[%(pathname)s:%(lineno)d %(funcName)s] - %(message)s"
    )
    formatter = logging.Formatter(fmt)

    logger = cutlass_logger.log()
    logger.setLevel(level=log_level)

    if log_to_console:
        console_handler = logging.StreamHandler()
        # console_handler.setLevel(log_level)
        # console_handler.setFormatter(formatter)
        logger.addHandler(console_handler)

    if log_to_file:
        from logging import FileHandler
        from datetime import datetime

        os.makedirs(logdir, exist_ok=True)

        dt = datetime.now().strftime("%Y%m%d_%H%M")
        new_handler = FileHandler(os.path.join(logdir, f"{dt}.log"), mode="w")
        # new_handler.setFormatter(formatter)
        # new_handler.setLevel(log_level)
        logger.addHandler(new_handler)

    for handler in logger.handlers:
        handler.setFormatter(formatter)
        handler.setLevel(log_level)

    return logger


if __name__ == "__main__":
    set_logging(log_to_console=True)
    patch_cutlass_loggers()
    import cutlass.cute as cute
