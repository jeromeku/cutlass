import os

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

def patch_cutlass_loggers():
    import logging
    import cutlass.base_dsl.utils.logger as cutlass_logger
    import cutlass
    
    fmt = ("%(asctime)s - %(name)s - %(levelname)s - "
           "[%(pathname)s:%(lineno)d %(funcName)s] - %(message)s")
    formatter = logging.Formatter(fmt)

    # Force the module to rebuild its logger using the new formatter.
    cutlass_logger.log = cutlass_logger.setup_log  # optional: make sure we can call it
    logger = cutlass_logger.setup_log(
        cutlass_logger.logger.name if cutlass_logger.logger else "generic",
        log_to_console=True,   # or False—match your needs
        log_to_file=False,
        log_level=logging.INFO,
    )

    for handler in logger.handlers:
        handler.setFormatter(formatter)

    return logger

if __name__ == "__main__":
    set_logging(log_to_console=True)
    patch_cutlass_loggers()
    import cutlass.cute as cute