import pygame
import random
import torch
import torch.nn as nn
import torch.optim as optim
import numpy as np

pygame.init()

# Set up the window
window = pygame.display.set_mode((500, 500))
pygame.display.set_caption("Pong with Q-Learning AI vs AI")

# Define game variables
paddle_width = 10
paddle_height = 100
ball_radius = 10
paddle_speed = 5

# Define Q-learning parameters
GAMMA = 0.99
LEARNING_RATE = 0.001
EPSILON = 0.1
BATCH_SIZE = 64
MEMORY_SIZE = 10000

# Define the Q-Network
class QNetwork(nn.Module):
    def __init__(self, input_size, output_size):
        super(QNetwork, self).__init__()
        self.fc1 = nn.Linear(input_size, 64)
        self.fc2 = nn.Linear(64, 32)
        self.fc3 = nn.Linear(32, output_size)

    def forward(self, x):
        x = torch.relu(self.fc1(x))
        x = torch.relu(self.fc2(x))
        return self.fc3(x)

# Initialize Q-Network and optimizer
q_network = QNetwork(5, 3)  # 5 input features, 3 actions (up, down, stay)
optimizer = optim.Adam(q_network.parameters(), lr=LEARNING_RATE)

# Initialize replay memory
replay_memory = []

# Helper functions
def get_state(py1, ball_x, ball_y, ball_dx, ball_dy):
    return torch.tensor([py1 / 500, ball_x / 500, ball_y / 500, ball_dx / 10, ball_dy / 10], dtype=torch.float32)

def choose_action(state, epsilon):
    if random.random() < epsilon:
        return random.randint(0, 2)
    else:
        with torch.no_grad():
            q_values = q_network(state)
            return torch.argmax(q_values).item()

def update_q_network():
    if len(replay_memory) < BATCH_SIZE:
        return

    batch = random.sample(replay_memory, BATCH_SIZE)
    states, actions, rewards, next_states, dones = zip(*batch)
    # the above zip is used to convert a list of tuples into a list of lists
    # otherwise we need manual unpack it
    states = torch.stack(states)
    next_states = torch.stack(next_states)
    """
    torch.stack()
    It takes a list (or a sequence) of tensors and stacks 
    them into a single tensor along a new axis, which makes it useful for 
    organizing data in batches.
    """
    actions = torch.tensor(actions)
    rewards = torch.tensor(rewards, dtype=torch.float32)
    dones = torch.tensor(dones, dtype=torch.float32)# dones is a tensor of bool values 
    # (true if the game is over)
    # 0 for not done, 1 for done and for computational efficiency float is used

    current_q_values = q_network(states).gather(1, actions.unsqueeze(1))
    """The gather(dim, index) function selects values from the input tensor 
        along a specific dimension
        
        
    """
    next_q_values = q_network(next_states).max(1)[0].detach()
    target_q_values = rewards + (1 - dones) * GAMMA * next_q_values

    loss = nn.MSELoss()(current_q_values.squeeze(), target_q_values)
    optimizer.zero_grad()
    loss.backward()
    optimizer.step()

# Game variables
py1 = 200  # Player 1 (left paddle) position
py2 = 200  # Player 2 (right paddle) position
ball_x, ball_y = 250, 250  # Ball initial position
ball_dx = random.choice([-4, 4])  # Ball horizontal speed
ball_dy = random.choice([-4, 4])  # Ball vertical speed
score1, score2 = 0, 0  # Scores

clock = pygame.time.Clock()

# Game loop
running = True
while running:
    # Fill window background with black
    window.fill((0, 0, 0))

    # Handle events
    for event in pygame.event.get():
        if event.type == pygame.QUIT:
            running = False

    # Get current state
    state = get_state(py1, ball_x, ball_y, ball_dx, ball_dy)

    # Choose action for Player 1 (Q-learning agent)
    action = choose_action(state, EPSILON)

    # Apply action
    if action == 0:  # Move up
        py1 -= paddle_speed
    elif action == 1:  # Move down
        py1 += paddle_speed
    # Action 2 is "stay"

    # Ensure paddle stays within bounds
    py1 = max(0, min(py1, 500 - paddle_height))

    # Simple AI for Player 2
    if ball_y > py2 + paddle_height / 2:
        py2 += paddle_speed
    elif ball_y < py2 + paddle_height / 2:
        py2 -= paddle_speed
    py2 = max(0, min(py2, 500 - paddle_height))

    # Ball movement
    ball_x += ball_dx
    ball_y += ball_dy

    # Ball collision with top and bottom walls
    if ball_y - ball_radius <= 0 or ball_y + ball_radius >= 500:
        ball_dy = -ball_dy

    # Ball collision with paddles
    if (ball_x - ball_radius <= 10 and py1 <= ball_y <= py1 + paddle_height) or \
       (ball_x + ball_radius >= 480 and py2 <= ball_y <= py2 + paddle_height):
        ball_dx = -ball_dx

    # Ball out of bounds (left or right side)
    done = False
    reward = 0
    if ball_x - ball_radius <= 0:
        score2 += 1
        ball_x, ball_y = 250, random.randrange(150,300)
        ball_dx, ball_dy = random.choice([-4, 4]), random.choice([-4, 4])
        reward = -1
        done = True
    if ball_x + ball_radius >= 500:
        score1 += 1
        ball_x, ball_y = 250, random.randrange(150,300)
        ball_dx, ball_dy = random.choice([-4, 4]), random.choice([-4, 4])
        reward = 1
        done = True

    # Get next state
    next_state = get_state(py1, ball_x, ball_y, ball_dx, ball_dy)

    # Store experience in replay memory
    replay_memory.append((state, action, reward, next_state, done))
    if len(replay_memory) > MEMORY_SIZE:
        replay_memory.pop(0)

    # Update Q-network
    update_q_network()

    # Drawing paddles, ball, and scores
    pygame.draw.rect(window, 'white', pygame.Rect(5, py1, paddle_width, paddle_height))  # Left paddle
    pygame.draw.rect(window, 'white', pygame.Rect(500 - 15, py2, paddle_width, paddle_height))  # Right paddle
    pygame.draw.circle(window, 'white', (int(ball_x), int(ball_y)), ball_radius)  # Ball

    # Draw scores
    font = pygame.font.SysFont(None, 50)
    score_text1 = font.render(f"{score1}", True, 'white')
    score_text2 = font.render(f"{score2}", True, 'white')
    window.blit(score_text1, (200, 20))
    window.blit(score_text2, (300, 20))

    # Update display
    pygame.display.update()
    clock.tick(60)

# Quit Pygame
pygame.quit()