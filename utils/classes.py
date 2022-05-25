class Aruco:
    def __init__(self, corners, id) -> None:
        self.topLeft, self.topRight, self.bottomRight, self.bottomLeft = corners
        self.id = id

    @property
    def corners(self):
        return self.topLeft, self.topRight, self.bottomRight, self.bottomLeft
