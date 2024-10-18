import pygame
import random
import torch
import torch.nn as nn
import torch.optim as optim
import numpy as np
import os

pygame.init()
WINDOW_WIDTH = 500
WINDOW_HEIGHT = 500
window = pygame.display.set_mode((WINDOW_WIDTH, WINDOW_HEIGHT))
pygame.display.set_caption("Pong with Q-Learning")

paddle_width = 10
paddle_height = 100
ball_radius = 10
paddle_speed = 5

#Q-learning parameters
GAMMA = 0.99
LEARNING_RATE = 0.001
EPSILON = 0.3
BATCH_SIZE = 64
MEMORY_SIZE = 10000
EPISODES = 400

device = torch.device("cuda" if torch.cuda.is_available() else "cpu") # Check for CUDA availability
print(f"Using device: {device}")

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
q_network = QNetwork(5, 3).to(device)  # 5 input features, 3 actions (up, down, stay)
optimizer = optim.Adam(q_network.parameters(), lr=LEARNING_RATE)

replay_memory = []

def get_state(py1, ball_x, ball_y, ball_dx, ball9_dy):
    return torch.tensor([py1 / WINDOW_HEIGHT, ball_x / WINDOW_WIDTH, ball_y / WINDOW_HEIGHT, ball_dx / 10, ball_dy / 10], dtype=torch.float32, device=device)

def choose_action(state, epsilon):
    if random.random() < epsilon:
        return random.randint(0, 2)
    else:
        with torch.no_grad():
            q_values = q_network(state) # predict action values for given state
            return torch.argmax(q_values).item()

def update_q_network():
    if len(replay_memory) < BATCH_SIZE:
        return

    batch = random.sample(replay_memory, BATCH_SIZE)
    states, actions, rewards, next_states, dones = zip(*batch)

    states = torch.stack(states)
    next_states = torch.stack(next_states)
    actions = torch.tensor(actions, device=device)
    rewards = torch.tensor(rewards, dtype=torch.float32, device=device)
    dones = torch.tensor(dones, dtype=torch.float32, device=device)

    current_q_values = q_network(states).gather(1, actions.unsqueeze(1))
    next_q_values = q_network(next_states).max(1)[0].detach()
    target_q_values = rewards + (1 - dones) * GAMMA * next_q_values

    loss = nn.MSELoss()(current_q_values.squeeze(), target_q_values)
    optimizer.zero_grad()
    loss.backward()
    optimizer.step()

def draw_game(py1, py2, ball_x, ball_y, score1, score2, episode, training=True):
    window.fill((0, 0, 0))

    # Draw paddles
    pygame.draw.rect(window, 'white', pygame.Rect(5, py1, paddle_width, paddle_height))
    pygame.draw.rect(window, 'white', pygame.Rect(WINDOW_WIDTH - 15, py2, paddle_width, paddle_height))

    # Draw ball
    pygame.draw.circle(window, 'white', (int(ball_x), int(ball_y)), ball_radius)

    # Draw scores
    font = pygame.font.SysFont(None, 50)
    score_text1 = font.render(f"{score1}", True, 'white')
    score_text2 = font.render(f"{score2}", True, 'white')
    window.blit(score_text1, (WINDOW_WIDTH // 4, 20))
    window.blit(score_text2, (3 * WINDOW_WIDTH // 4, 20))

    # Draw episode number if training
    if training:
        episode_text = font.render(f"Episode: {episode + 1}", True, 'white')
        window.blit(episode_text, (WINDOW_WIDTH // 2 - 70, WINDOW_HEIGHT - 40))

    pygame.display.update()

def train_ai(visible_training=True):
    global py1, py2, ball_x, ball_y, ball_dx, ball_dy, score1, score2

    clock = pygame.time.Clock()

    for episode in range(EPISODES):
        py1, py2 = WINDOW_HEIGHT // 2 - paddle_height // 2, WINDOW_HEIGHT // 2 - paddle_height // 2
        ball_x, ball_y = WINDOW_WIDTH // 2, WINDOW_HEIGHT // 2
        ball_dx = random.choice([-4, 4])
        ball_dy = random.choice([-4, 4])
        score1, score2 = 0, 0

        while True:
            if visible_training:
                for event in pygame.event.get():
                    if event.type == pygame.QUIT:
                        pygame.quit()
                        return

            state = get_state(py1, ball_x, ball_y, ball_dx, ball_dy)
            action = choose_action(state, EPSILON)

            if action == 0:  # Move up
                py1 -= paddle_speed
            elif action == 1:  # Move down
                py1 += paddle_speed
            py1 = max(0, min(py1, WINDOW_HEIGHT - paddle_height))

            # Simple AI for Player 2
            if ball_y > py2 + paddle_height // 2:
                py2 += paddle_speed
            elif ball_y < py2 + paddle_height // 2:
                py2 -= paddle_speed
            py2 = max(0, min(py2, WINDOW_HEIGHT - paddle_height))

            ball_x += ball_dx
            ball_y += ball_dy

            # Ball collision with top and bottom walls
            if ball_y - ball_radius <= 0 or ball_y + ball_radius >= WINDOW_HEIGHT:
                ball_dy = -ball_dy

            # Create Rect objects for collision detection
            ball_rect = pygame.Rect(ball_x - ball_radius, ball_y - ball_radius, ball_radius * 2, ball_radius * 2)
            paddle1_rect = pygame.Rect(5, py1, paddle_width, paddle_height)
            paddle2_rect = pygame.Rect(WINDOW_WIDTH - 15, py2, paddle_width, paddle_height)

            # Ball collision with paddles
            if ball_rect.colliderect(paddle1_rect) or ball_rect.colliderect(paddle2_rect):
                ball_dx = -ball_dx

            done = False
            reward = 0
            if ball_x - ball_radius <= 0:
                score2 += 1
                ball_x, ball_y = WINDOW_WIDTH // 2, WINDOW_HEIGHT // 2
                ball_dx, ball_dy = random.choice([-4, 4]), random.choice([-4, 4])
                reward = -1
                done = True
            if ball_x + ball_radius >= WINDOW_WIDTH:
                score1 += 1
                ball_x, ball_y = WINDOW_WIDTH // 2, WINDOW_HEIGHT // 2
                ball_dx, ball_dy = random.choice([-4, 4]), random.choice([-4, 4])
                reward = 1
                done = True

            next_state = get_state(py1, ball_x, ball_y, ball_dx, ball_dy)

            replay_memory.append((state, action, reward, next_state, done))
            if len(replay_memory) > MEMORY_SIZE:
                replay_memory.pop(0)

            update_q_network()

            if visible_training:
                draw_game(py1, py2, ball_x, ball_y, score1, score2, episode)
                clock.tick(60)

            if done:
                break

        if (episode + 1) % 10 == 0:
            print(f"Episode {episode + 1}/{EPISODES} completed. Score: {score1}-{score2}")

    print("Training complete. Saving model...")
    torch.save(q_network.state_dict(), "pong_ai_model.pth")

def play_vs_ai():
    global py1, py2, ball_x, ball_y, ball_dx, ball_dy, score1, score2

    py1, py2 = WINDOW_HEIGHT // 2 - paddle_height // 2, WINDOW_HEIGHT // 2 - paddle_height // 2
    ball_x, ball_y = WINDOW_WIDTH // 2, WINDOW_HEIGHT // 2
    ball_dx = random.choice([-4, 4])
    ball_dy = random.choice([-4, 4])
    score1, score2 = 0, 0

    clock = pygame.time.Clock()

    running = True
    while running:
        for event in pygame.event.get():
            if event.type == pygame.QUIT:
                running = False

        keys = pygame.key.get_pressed()
        if keys[pygame.K_w] and py1 > 0:
            py1 -= paddle_speed
        if keys[pygame.K_s] and py1 < WINDOW_HEIGHT - paddle_height:
            py1 += paddle_speed

        state = get_state(py2, WINDOW_WIDTH - ball_x, ball_y, -ball_dx, ball_dy)
        action = choose_action(state, 0)  # No exploration during gameplay

        if action == 0:  # Move up
            py2 -= paddle_speed
        elif action == 1:  # Move down
            py2 += paddle_speed
        py2 = max(0, min(py2, WINDOW_HEIGHT - paddle_height))

        ball_x += ball_dx
        ball_y += ball_dy

        # Ball collision with top and bottom walls
        if ball_y - ball_radius <= 0 or ball_y + ball_radius >= WINDOW_HEIGHT:
            ball_dy = -ball_dy

        # Create Rect objects for collision detection
        ball_rect = pygame.Rect(ball_x - ball_radius, ball_y - ball_radius, ball_radius * 2, ball_radius * 2)
        paddle1_rect = pygame.Rect(5, py1, paddle_width, paddle_height)
        paddle2_rect = pygame.Rect(WINDOW_WIDTH - 15, py2, paddle_width, paddle_height)

        # Ball collision with paddles
        if ball_rect.colliderect(paddle1_rect) or ball_rect.colliderect(paddle2_rect):
            ball_dx = -ball_dx

        if ball_x - ball_radius <= 0:
            score2 += 1
            ball_x, ball_y = WINDOW_WIDTH // 2, WINDOW_HEIGHT // 2
            ball_dx, ball_dy = random.choice([-4, 4]), random.choice([-4, 4])
        if ball_x + ball_radius >= WINDOW_WIDTH:
            score1 += 1
            ball_x, ball_y = WINDOW_WIDTH // 2, WINDOW_HEIGHT // 2
            ball_dx, ball_dy = random.choice([-4, 4]), random.choice([-4, 4])

        draw_game(py1, py2, ball_x, ball_y, score1, score2, 0, training=False)
        clock.tick(60)

    pygame.quit()

# Main execution
if __name__ == "__main__":
    if not os.path.exists("pong_ai_model.pth"): # checks the model exists or not
        print("Training AI...")
        visible_training = input("Do you want to see the training process? (y/n): ").lower() == 'y'
        train_ai(False)
    else:
        print("Loading pre-trained model...")
        q_network.load_state_dict(torch.load("pong_ai_model.pth", map_location=device))

    print("Starting game: Player vs AI")
    play_vs_ai()