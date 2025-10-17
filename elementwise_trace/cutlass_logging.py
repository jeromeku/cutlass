import os
import logging

def set_logging(log_to_console: bool = False, log_level: int = 10, log_to_file: str = None):
    if log_to_console:
        os.environ["CUTE_DSL_LOG_TO_CONSOLE"] = "1"
    if log_to_file is not None:
        os.environ["CUTE_DSL_LOG_TO_FILE"] = log_to_file
    # need to patch logger cutlass.base_dsl.utils.logger formatting
#    formatter = logging.Formatter(
        # fmt="%(asctime)s %(name)s %(levelname)s %(pathname)s:%(lineno)d [%(funcName)s] %(message)s",
        # datefmt="%Y-%m-%d %H:%M:%S",
#    )

    os.environ["CUTE_DSL_LOG_LEVEL"] = str(log_level)

def set_compilation_opts(dry_run: bool = False, dump_ptx: bool = False, ptx_dir: str = "cute_dsl_ptx", print_ir: bool = False, keep_ir: bool = False):
    if print_ir:
        os.environ["CUTE_DSL_PRINT_IR"] = "1"
    if keep_ir:
        os.environ["CUTE_DSL_KEEP_IR"] = "1"
    if dump_ptx:
        os.environ["CUTE_DSL_PTX"] = "1"
        os.environ["CUTE_DSL_DUMP_PTX"] = "1"
        os.environ["CUTE_DSL_PTX_DIR"] = ptx_dir
    if dry_run:
        os.environ["CUTE_DSL_DRYRUN"] = "1"
    
def set_arch(arch: str):
    from cutlass.base_dsl import detect_gpu_arch
    os.environ.setdefault("CUTE_DSL_ARCH", detect_gpu_arch(None))

def patch_cutlass_loggers(log_to_console: bool = False, log_to_file: bool = False, logdir: str = "cute_logs", log_level: int = logging.DEBUG):
    
    import cutlass.base_dsl.utils.logger as cutlass_logger
    import os

    log_to_file = log_to_file or os.environ.get("CUTE_DSL_LOG_TO_FILE", "0") == "1"
    log_to_console = log_to_console or os.environ.get("CUTE_DSL_LOG_TO_CONSOLE", "0") == "1"

    if not (log_to_console or log_to_file):
        return
    
    fmt = ("%(asctime)s - %(name)s - %(levelname)s - "
           "[%(pathname)s:%(lineno)d %(funcName)s] - %(message)s")
    formatter = logging.Formatter(fmt)
    
    logger = cutlass_logger.log()
    if log_to_file:
        from logging import FileHandler
        from datetime import datetime

        os.makedirs(logdir, exist_ok=True)

        dt = datetime.now().strftime("%Y%m%d_%H%M")
        new_handler = FileHandler(os.path.join(logdir, f'{dt}.log'), mode = 'w')
        new_handler.setFormatter(formatter)
        new_handler.setLevel(log_level)
        logger.addHandler(new_handler)

    for handler in logger.handlers:
        handler.setFormatter(formatter)
        handler.setLevel(log_level)
    
    return logger

if __name__ == "__main__":
    set_logging(log_to_console=True)
    patch_cutlass_loggers()
    import cutlass.cute as cute