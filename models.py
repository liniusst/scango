class User:
    def __init__(self, id, is_active):
        self.id = id
        self._is_active = is_active

    def is_authenticated(self):
        return self._is_active

    def is_active(self):
        return self._is_active

    def is_anonymous(self):
        return False

    def get_id(self):
        return str(self.id)

    @property
    def is_active(self):
        return self._is_active

    @is_active.setter
    def is_active(self, value):
        self._is_active = value
