import json
from pathlib import Path


class ConfigNamespace:
    """
    Hilfsklasse: erlaubt Zugriff auf Dicts auch per Attribut.
    Beispiel:
        settings.preprocessing.label_normalize
        statt settings["preprocessing"]["label_normalize"]
    """
    def __init__(self, d: dict):
        for k, v in d.items():
            if isinstance(v, dict):
                v = ConfigNamespace(v)
            setattr(self, k, v)

    def as_dict(self) -> dict:
        """Gibt die Config als normales Dict zurück (rekursiv aufgelöst)."""
        return {k: (v.as_dict() if isinstance(v, ConfigNamespace) else v)
                for k, v in self.__dict__.items()}

    def __getitem__(self, item):
        return getattr(self, item)


class Settings(ConfigNamespace):
    """Lädt die JSON-Config und stellt sie als Objektstruktur bereit."""
    def __init__(self, path: str | Path):
        path = Path(path)
        if not path.exists():
            raise FileNotFoundError(f"Config file not found: {path}")

        with open(path, "r") as f:
            cfg = json.load(f)

        super().__init__(cfg)



