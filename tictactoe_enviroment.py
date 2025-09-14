class Enviroment:
    def __init__(self):
        self.grid = [
            0, 0, 0,
            0, 0, 0,
            0, 0, 0
        ]
        self.last_input = [0, 0]

    def check_game_end(self):
        end = 0
        winner = self.last_input[1]
        if self.grid[0] == self.grid[1] == self.grid[2] != 0:
            end = 1
        elif self.grid[3] == self.grid[4] == self.grid[5] != 0:
            end = 1
        elif self.grid[6] == self.grid[7] == self.grid[8] != 0:
            end = 1
        elif self.grid[0] == self.grid[3] == self.grid[6] != 0:
            end = 1
        elif self.grid[1] == self.grid[4] == self.grid[7] != 0:
            end = 1
        elif self.grid[2] == self.grid[5] == self.grid[8] != 0:
            end = 1
        elif self.grid[0] == self.grid[4] == self.grid[8] != 0:
            end = 1
        elif self.grid[2] == self.grid[4] == self.grid[6] != 0:
            end = 1
        return [end, winner]
    
    def get_state(self):
        game_end = self.check_game_end()
        return [self.grid, game_end]
    
    def input(self, input_pla):
        try:
            if self.grid[input_pla[0]] == 0:
                self.grid[input_pla[0]] = input_pla[1]
                self.last_input = input_pla
            else:
                return "Illegal Move"
        except:
            return "Error"
        
test = Enviroment()
