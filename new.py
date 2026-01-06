import numpy as np
import pygame
import matplotlib.pyplot as plt
from collections import deque
import random

# Environment Configuration
class MazeConfig:
    WIDTH = 600
    HEIGHT = 600
    CELL_SIZE = 60
    FPS = 60
    
    # Colors
    BLACK = (0, 0, 0)
    WHITE = (255, 255, 255)
    RED = (255, 0, 0)
    GREEN = (0, 255, 0)
    BLUE = (0, 0, 255)
    GRAY = (128, 128, 128)
    YELLOW = (255, 255, 0)

class Agent:
    def __init__(self, x, y, radius=15):
        self.x = x
        self.y = y
        self.vx = 0
        self.vy = 0
        self.radius = radius
        self.stunned = False
        self.stun_timer = 0
        self.stun_duration = 120  # frames (2 seconds at 60 FPS)
        
    def update(self, dvx, dvy):
        """Update velocity based on action"""
        if self.stunned:
            # Reduce control when stunned
            dvx *= 0.1
            dvy *= 0.1
            self.stun_timer -= 1
            if self.stun_timer <= 0:
                self.stunned = False
        
        # Apply velocity change with limits
        self.vx += dvx
        self.vy += dvy
        
        # Velocity limits
        max_vel = 5.0
        self.vx = np.clip(self.vx, -max_vel, max_vel)
        self.vy = np.clip(self.vy, -max_vel, max_vel)
        
        # Apply friction
        self.vx *= 0.95
        self.vy *= 0.95
        
        # Update position
        self.x += self.vx
        self.y += self.vy
    
    def stun(self):
        """Apply stun effect"""
        self.stunned = True
        self.stun_timer = self.stun_duration
        self.vx *= 0.3
        self.vy *= 0.3

class MazeEnvironment:
    def __init__(self):
        self.width = MazeConfig.WIDTH
        self.height = MazeConfig.HEIGHT
        self.agent = Agent(50, 50)
        self.goal = [550, 550]
        self.walls = self._create_maze()
        self.max_steps = 1000
        self.current_step = 0
        
    def _create_maze(self):
        """Create a simple maze with walls"""
        walls = []
        cell = MazeConfig.CELL_SIZE
        
        # Outer walls
        walls.append(pygame.Rect(0, 0, self.width, 10))  # Top
        walls.append(pygame.Rect(0, self.height-10, self.width, 10))  # Bottom
        walls.append(pygame.Rect(0, 0, 10, self.height))  # Left
        walls.append(pygame.Rect(self.width-10, 0, 10, self.height))  # Right
        
        # Internal walls
        walls.append(pygame.Rect(cell*2, 0, 10, cell*5))
        walls.append(pygame.Rect(cell*4, cell*3, 10, cell*7))
        walls.append(pygame.Rect(cell*6, 0, 10, cell*6))
        walls.append(pygame.Rect(cell*1, cell*6, cell*3, 10))
        walls.append(pygame.Rect(cell*5, cell*7, cell*4, 10))
        
        return walls
    
    def get_state(self):
        """Get agent's sensor readings"""
        # Distance sensors in 4 directions (normalized 0-1)
        sensors = {
            'top': self._cast_ray(0, -1),
            'left': self._cast_ray(-1, 0),
            'down': self._cast_ray(0, 1),
            'right': self._cast_ray(1, 0),
        }
        
        # Distance to goal (normalized)
        dx = self.goal[0] - self.agent.x
        dy = self.goal[1] - self.agent.y
        dist_to_goal = np.sqrt(dx**2 + dy**2) / np.sqrt(self.width**2 + self.height**2)
        
        # Stun flag
        stun_flag = 1.0 if self.agent.stunned else 0.0
        
        return np.array([
            sensors['top'],
            sensors['left'],
            sensors['down'],
            sensors['right'],
            dist_to_goal,
            stun_flag
        ])
    
    def _cast_ray(self, dx, dy, max_dist=200):
        """Cast a ray to detect walls"""
        x, y = self.agent.x, self.agent.y
        dist = 0
        
        while dist < max_dist:
            x += dx * 2
            y += dy * 2
            dist += 2
            
            # Check wall collision
            for wall in self.walls:
                if wall.collidepoint(x, y):
                    return 1.0 - (dist / max_dist)
        
        return 0.0
    
    def step(self, action):
        """Execute action and return state, reward, done"""
        dvx, dvy = action
        self.agent.update(dvx, dvy)
        self.current_step += 1
        
        # Check wall collision
        agent_rect = pygame.Rect(
            self.agent.x - self.agent.radius,
            self.agent.y - self.agent.radius,
            self.agent.radius * 2,
            self.agent.radius * 2
        )
        
        collision = False
        for wall in self.walls:
            if agent_rect.colliderect(wall):
                collision = True
                # Push agent out
                if abs(self.agent.vx) > abs(self.agent.vy):
                    self.agent.x -= self.agent.vx * 10
                else:
                    self.agent.y -= self.agent.vy * 10
                self.agent.stun()
                break
        
        # Calculate reward
        reward = self._calculate_reward(collision)
        
        # Check if goal reached
        dx = self.goal[0] - self.agent.x
        dy = self.goal[1] - self.agent.y
        dist = np.sqrt(dx**2 + dy**2)
        done = dist < 30 or self.current_step >= self.max_steps
        
        if dist < 30:
            reward += 100  # Big reward for reaching goal
        
        return self.get_state(), reward, done
    
    def _calculate_reward(self, collision):
        """Calculate reward based on current state"""
        dx = self.goal[0] - self.agent.x
        dy = self.goal[1] - self.agent.y
        dist = np.sqrt(dx**2 + dy**2)
        
        # Reward for moving closer to goal
        reward = -dist / 1000
        
        # Penalty for collision
        if collision:
            reward -= 5
        
        # Small time penalty
        reward -= 0.01
        
        return reward
    
    def reset(self):
        """Reset environment"""
        self.agent = Agent(50, 50)
        self.current_step = 0
        return self.get_state()

class NeuralNetwork:
    def __init__(self, input_size, hidden_size, output_size):
        self.w1 = np.random.randn(input_size, hidden_size) * 0.5
        self.b1 = np.zeros(hidden_size)
        self.w2 = np.random.randn(hidden_size, hidden_size) * 0.5
        self.b2 = np.zeros(hidden_size)
        self.w3 = np.random.randn(hidden_size, output_size) * 0.5
        self.b3 = np.zeros(output_size)
    
    def forward(self, x):
        """Forward pass"""
        h1 = np.tanh(np.dot(x, self.w1) + self.b1)
        h2 = np.tanh(np.dot(h1, self.w2) + self.b2)
        output = np.tanh(np.dot(h2, self.w3) + self.b3)
        return output
    
    def get_params(self):
        """Get all network parameters"""
        return [self.w1, self.b1, self.w2, self.b2, self.w3, self.b3]
    
    def set_params(self, params):
        """Set all network parameters"""
        self.w1, self.b1, self.w2, self.b2, self.w3, self.b3 = params

class EvolutionaryTrainer:
    def __init__(self, population_size=20):
        self.population_size = population_size
        self.population = [NeuralNetwork(6, 16, 2) for _ in range(population_size)]
        self.fitness_history = []
        
    def evaluate(self, network, env, render=False, screen=None):
        """Evaluate a network's performance"""
        state = env.reset()
        total_reward = 0
        done = False
        
        while not done:
            if render and screen:
                self._render(env, screen)
            
            action = network.forward(state)
            state, reward, done = env.step(action)
            total_reward += reward
        
        return total_reward
    
    def _render(self, env, screen):
        """Render the environment"""
        screen.fill(MazeConfig.WHITE)
        
        # Draw walls
        for wall in env.walls:
            pygame.draw.rect(screen, MazeConfig.BLACK, wall)
        
        # Draw goal
        pygame.draw.circle(screen, MazeConfig.GREEN, env.goal, 20)
        
        # Draw agent
        color = MazeConfig.YELLOW if env.agent.stunned else MazeConfig.BLUE
        pygame.draw.circle(screen, color, 
                         (int(env.agent.x), int(env.agent.y)), 
                         env.agent.radius)
        
        pygame.display.flip()
        pygame.time.Clock().tick(MazeConfig.FPS)
    
    def evolve(self, fitness_scores):
        """Evolve population based on fitness"""
        # Select top performers
        indices = np.argsort(fitness_scores)[::-1]
        elite_count = self.population_size // 5
        
        new_population = []
        
        # Keep elite
        for i in range(elite_count):
            new_population.append(self.population[indices[i]])
        
        # Create offspring
        while len(new_population) < self.population_size:
            parent1 = self.population[indices[np.random.randint(elite_count)]]
            parent2 = self.population[indices[np.random.randint(elite_count)]]
            child = self._crossover(parent1, parent2)
            self._mutate(child)
            new_population.append(child)
        
        self.population = new_population
    
    def _crossover(self, parent1, parent2):
        """Create offspring from two parents"""
        child = NeuralNetwork(6, 16, 2)
        p1_params = parent1.get_params()
        p2_params = parent2.get_params()
        child_params = []
        
        for p1, p2 in zip(p1_params, p2_params):
            mask = np.random.rand(*p1.shape) > 0.5
            child_params.append(np.where(mask, p1, p2))
        
        child.set_params(child_params)
        return child
    
    def _mutate(self, network, mutation_rate=0.1):
        """Mutate network parameters"""
        params = network.get_params()
        for i in range(len(params)):
            mask = np.random.rand(*params[i].shape) < mutation_rate
            params[i] += mask * np.random.randn(*params[i].shape) * 0.5
        network.set_params(params)
    
    def train(self, generations=500, visualize_best=True):
        """Train the population"""
        env = MazeEnvironment()
        
        # Setup pygame for visualization
        if visualize_best:
            pygame.init()
            screen = pygame.display.set_mode((MazeConfig.WIDTH, MazeConfig.HEIGHT))
            pygame.display.set_caption("Maze RL Training")
        else:
            screen = None
        
        for gen in range(generations):
            fitness_scores = []
            
            # Evaluate population
            for i, network in enumerate(self.population):
                render = visualize_best and i == 0 and gen % 5 == 0
                fitness = self.evaluate(network, env, render, screen)
                fitness_scores.append(fitness)
                
                # Handle pygame events
                if visualize_best:
                    for event in pygame.event.get():
                        if event.type == pygame.QUIT:
                            pygame.quit()
                            return
            
            # Record best fitness
            best_fitness = max(fitness_scores)
            avg_fitness = np.mean(fitness_scores)
            self.fitness_history.append((best_fitness, avg_fitness))
            
            print(f"Generation {gen+1}: Best={best_fitness:.2f}, Avg={avg_fitness:.2f}")
            
            # Evolve
            self.evolve(fitness_scores)
        
        if visualize_best:
            pygame.quit()
        
        # Plot results
        self.plot_results()
    
    def plot_results(self):
        """Plot training progress"""
        best_scores = [x[0] for x in self.fitness_history]
        avg_scores = [x[1] for x in self.fitness_history]
        
        plt.figure(figsize=(10, 6))
        plt.plot(best_scores, label='Best Fitness', linewidth=2)
        plt.plot(avg_scores, label='Average Fitness', linewidth=2)
        plt.xlabel('Generation')
        plt.ylabel('Fitness')
        plt.title('Training Progress')
        plt.legend()
        plt.grid(True, alpha=0.3)
        plt.tight_layout()
        plt.show()

# Run training
if __name__ == "__main__":
    trainer = EvolutionaryTrainer(population_size=20)
    trainer.train(generations=50, visualize_best=True)