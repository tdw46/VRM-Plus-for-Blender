# pylint: disable=c-extension-no-member, unused-argument

class IDPropertyGroup:
    def clear(self) -> None: ...
    def get(self, key: str, default: object = None) -> object: ...