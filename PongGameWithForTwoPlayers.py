import pygame
import random

# Initialize Pygame
pygame.init()

# Set up the window
window = pygame.display.set_mode((500, 500))
pygame.display.set_caption("Pong")

# Define game variables
py1 = 200  # Player 1 (left paddle) position
py2 = 200  # Player 2 (right paddle) position
paddle_speed = 5

ball_x = 250  # Ball initial position (x-axis)
ball_y = 250  # Ball initial position (y-axis)
ball_dx = random.choice([-4, 4])  # Ball horizontal speed
ball_dy = random.choice([-4, 4])  # Ball vertical speed

paddle_width = 10
paddle_height = 100
ball_radius = 10

score1 = 0  # Score for Player 1
score2 = 0  # Score for Player 2

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

    # Get key presses for player movement
    keys = pygame.key.get_pressed()
    
    # Player 1 (W/S keys)
    if keys[pygame.K_w]:
        py1 -= paddle_speed
        if py1 < 0:
            py1 = 0
    if keys[pygame.K_s]:
        py1 += paddle_speed
        if py1 > 500 - paddle_height:
            py1 = 500 - paddle_height
    
    # Player 2 (Up/Down arrow keys)
    if keys[pygame.K_UP]:
        py2 -= paddle_speed
        if py2 < 0:
            py2 = 0
    if keys[pygame.K_DOWN]:
        py2 += paddle_speed
        if py2 > 500 - paddle_height:
            py2 = 500 - paddle_height
    
    # Ball movement
    ball_x += ball_dx
    ball_y += ball_dy

    # Ball collision with top and bottom walls
    if ball_y - ball_radius <= 0 or ball_y + ball_radius >= 500:
        ball_dy = -ball_dy

    # Ball collision with paddles
    if (ball_x - ball_radius <= 10 and py1 <= ball_y <= py1 + paddle_height) or \
       (ball_x + ball_radius >= 485 and py2 <= ball_y <= py2 + paddle_height):
        ball_dx = -ball_dx
    
    # Ball out of bounds (left or right side)
    if ball_x - ball_radius <= 0:
        score2 += 1
        ball_x, ball_y = 250, random.randrange(150,300)
        ball_dx, ball_dy = random.choice([-4, 4]), random.choice([-4, 4])
    if ball_x + ball_radius >= 500:
        score1 += 1
        ball_x, ball_y = 250, random.randrange(150,300)
        ball_dx, ball_dy = random.choice([-4, 4]), random.choice([-4, 4])

    # Drawing paddles, ball, and scores
    pygame.draw.rect(window, 'white', pygame.Rect(5, py1, paddle_width, paddle_height))  # Left paddle
    pygame.draw.rect(window, 'white', pygame.Rect(500 - 15, py2, paddle_width, paddle_height))  # Right paddle
    pygame.draw.circle(window, 'white', (ball_x, ball_y), ball_radius)  # Ball

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
