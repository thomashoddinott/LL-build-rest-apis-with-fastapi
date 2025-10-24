from dynaconf import Dynaconf

settings = Dynaconf(
    envvar_prefix="DEV",
    settings_files=["settings.yaml", ".secrets.yaml"],
)
