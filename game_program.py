import pygame
from pygame.locals import *
from sys import exit

#
pygame.init()
# Set up the display window
screen = pygame.display.set_mode((640, 480), 0, 32)
pygame.display.set_caption("Pong Pong!")

# Create surfaces for the background, paddles, and ball
background = pygame.Surface((640, 480))
background.fill((0, 0, 0))  # Black background

# Paddle surfaces
bar = pygame.Surface((10, 50))  # Paddle dimensions
bar1 = bar.convert()
bar1.fill((0, 0, 255))  # Blue paddle for Player 1
bar2 = bar.convert()
bar2.fill((0, 255, 0))  # Green paddle for Player 2

# Ball surface
circ_sur = pygame.Surface((15, 15))
pygame.draw.circle(circ_sur, (255, 0, 0), (15 // 2, 15 // 2), 15 // 2)  # Red ball
circle = circ_sur.convert()
circle.set_colorkey((0, 0, 0))  # Make the ball background transparent

# Initial positions and movement speeds
bar1_x, bar2_x = 10., 620.  # X positions for paddles
bar1_y, bar2_y = 215., 215.  # Starting Y positions for paddles
circle_x, circle_y = 307.5, 232.5  # Ball starting position
speed_x, speed_y = 250., 250.  # Ball speed in pixels per second
bar1_score, bar2_score = 0, 0  # Player scores

# Setup for frame rate control and font rendering
clock = pygame.time.Clock()
font = pygame.font.SysFont("calibri", 40)

# Main game loop

while True:
    # Handle events
    for event in pygame.event.get():
        if event.type == QUIT:
            exit()

    # Render the current scores
    score1 = font.render(str(bar1_score), True, (255, 255, 255))
    score2 = font.render(str(bar2_score), True, (255, 255, 255))

    # Draw the background, paddles, ball, and scores
    screen.blit(background, (0, 0))  # Clear the screen
    pygame.draw.rect(screen, (255, 255, 255), (5, 5, 630, 470), 2)  # Draw border
    pygame.draw.aaline(screen, (255, 255, 255), (330, 5), (330, 475))  # Center line
    screen.blit(bar1, (bar1_x, bar1_y))  # Draw Player 1's paddle
    screen.blit(bar2, (bar2_x, bar2_y))  # Draw Player 2's paddle
    screen.blit(circle, (circle_x, circle_y))  # Draw the ball
    screen.blit(score1, (250., 210.))  # Display Player 1's score
    screen.blit(score2, (380., 210.))  # Display Player 2's score

    # Update ball position based on its speed
    time_passed = clock.tick(30)  # Cap the frame rate to 30 FPS
    time_sec = time_passed / 1000.0  # Convert milliseconds to seconds

    circle_x += speed_x * time_sec
    circle_y += speed_y * time_sec

    # Collision detection with paddles
    if circle_x <= bar1_x + 10.:
        if bar1_y - 7.5 <= circle_y <= bar1_y + 42.5:
            circle_x = 20.
            speed_x = -speed_x  # Reverse X direction

    if circle_x >= bar2_x - 15.:
        if bar2_y - 7.5 <= circle_y <= bar2_y + 42.5:
            circle_x = 605.
            speed_x = -speed_x  # Reverse X direction

    # Ball goes out of bounds, update the score
    if circle_x < 5.:
        bar2_score += 1
        circle_x, circle_y = 320., 232.5  # Reset ball position
        bar1_y, bar2_y = 215., 215.  # Reset paddles

    elif circle_x > 620.:
        bar1_score += 1
        circle_x, circle_y = 307.5, 232.5  # Reset ball position
        bar1_y, bar2_y = 215., 215.  # Reset paddles

    # Ball collision with top and bottom of the screen
    if circle_y <= 10.:
        speed_y = -speed_y  # Reverse Y direction
        circle_y = 10.

    elif circle_y >= 457.5:
        speed_y = -speed_y  # Reverse Y direction
        circle_y = 457.5

    pygame.display.update()
    # Update the display with new drawings