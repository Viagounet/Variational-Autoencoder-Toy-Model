from dataclasses import dataclass, field
import math
from random import randrange
import pygame
from pygame import draw
import os

# pygame setup
pygame.init()
screen = pygame.display.set_mode((200, 200))

# Create the imgs/ directory if it doesn't exist
if not os.path.exists("imgs"):
    os.makedirs("imgs")


@dataclass
class Force:
    name: str
    vector: pygame.math.Vector2


frame_count = 0

for i in range(10):
    clock = pygame.time.Clock()
    running = True
    dt = 0

    # font = pygame.font.Font(None, 74)
    t = 0
    circle_init_pos_x = 20
    circle_init_pos_y = 10
    circle_pos = pygame.Vector2(circle_init_pos_x, circle_init_pos_y)
    rectange_init_pos_x = randrange(100, 190)
    rectangle_init_pos_y = randrange(100, 190)
    rectangle_pos = pygame.Vector2(rectange_init_pos_x, rectangle_init_pos_y)

    acc = 0
    speed = 0
    while running:
        for event in pygame.event.get():
            if event.type == pygame.QUIT:
                running = False

        # fill the screen with a color to wipe away anything from last frame
        screen.fill("purple")

        acc += dt / 10
        speed += acc
        circle = pygame.draw.circle(
            screen,
            pygame.Color(int(t * 60 * 3 % 255), 0, 0),
            circle_pos + pygame.Vector2(0, speed),
            15,
        )
        rect = pygame.draw.rect(
            screen,
            pygame.Color(50, 50, int(t * 60 * 3 % 255)),
            pygame.Rect(
                rectangle_pos.x, rectangle_pos.y, 10, int(10 + math.sin(dt) * 10)
            ),
        )

        # Update timer
        t += dt

        # Save the current frame as an image
        pygame.image.save(screen, f"imgs/frame_{frame_count:04d}.png")
        frame_count += 1

        # flip() the display to put your work on screen
        pygame.display.flip()

        # limits FPS to 60
        # dt is delta time in seconds since last frame, used for framerate-
        # independent physics.
        dt = clock.tick(60) / 1000
        if t > 5:
            running = False

pygame.quit()
