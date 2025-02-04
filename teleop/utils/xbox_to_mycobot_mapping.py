class XboxToMyCobotMapping:
    def __init__(self, nickname, key, scale, min_value=0.1, negative_key=None):
        self.nickname = nickname
        self.key = key
        self.scale = scale
        self.min_value = min_value
        self.negative_key = negative_key

    def parse(self, controller_state):
        raw_input = controller_state[self.key]
        if self.negative_key:
            raw_input = raw_input - controller_state[self.negative_key]
        if abs(raw_input) < self.min_value:
            return 0
        return self.scale * raw_input
    
    def __str__(self):
        return f'{self.nickname} {self.key} {self.scale}'
    
    def __repr__(self):
        return str(self)
    