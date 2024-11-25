import pygame
from pygame.locals import *
from sys import exit
import game_detectObject

# Inicializar o Pygame
pygame.init()

# Configuração da janela
screen = pygame.display.set_mode((640, 480), 0, 32)
pygame.display.set_caption("Pong Pong!")

# Configurações do jogo
background = pygame.Surface((640, 480))
background.fill((0, 0, 0))
bar = pygame.Surface((10, 50))
bar1 = bar.convert()
bar1.fill((0, 0, 255))
bar2 = bar.convert()
bar2.fill((0, 255, 0))

circ_sur = pygame.Surface((15, 15))
pygame.draw.circle(circ_sur, (255, 0, 0), (15 // 2, 15 // 2), 15 // 2)
circle = circ_sur.convert()
circle.set_colorkey((0, 0, 0))

# Posições e velocidade iniciais
bar1_x, bar2_x = 10., 620.
bar1_y, bar2_y = 215., 215.
circle_x, circle_y = 307.5, 232.5
speed_x, speed_y = 65., 65.
bar1_score, bar2_score = 0, 0

# Relógio e fonte
clock = pygame.time.Clock()
font = pygame.font.SysFont("calibri", 40)


# Função para atualizar o jogo
def update_game():
    global bar1_y, bar2_y, circle_x, circle_y, speed_x, speed_y, bar1_score, bar2_score

    # Capturar eventos
    for event in pygame.event.get():
        if event.type == QUIT:
            game_detectObject.release_resources()
            return False

    # Atualizar paddles com detecção
    center_blue, center_green = game_detectObject.update()
    if center_green:
        bar2_y = center_green - 25
    if center_blue:
        bar1_y = center_blue - 25
    # Restringir movimento dos paddles ao limite da tela
    bar1_y = max(10, min(430, bar1_y))
    bar2_y = max(10, min(430, bar2_y))

    # Renderizar placar
    score1 = font.render(str(bar1_score), True, (255, 255, 255))
    score2 = font.render(str(bar2_score), True, (255, 255, 255))

    # Desenhar elementos do jogo
    screen.blit(background, (0, 0))
    pygame.draw.rect(screen, (255, 255, 255), (5, 5, 630, 470), 2)
    pygame.draw.aaline(screen, (255, 255, 255), (330, 5), (330, 475))
    screen.blit(bar1, (bar1_x, bar1_y))
    screen.blit(bar2, (bar2_x, bar2_y))
    screen.blit(circle, (circle_x, circle_y))
    screen.blit(score1, (250., 210.))
    screen.blit(score2, (380., 210.))

    # Atualizar posição da bola
    time_passed = clock.tick(30)
    time_sec = time_passed / 1000.0
    circle_x += speed_x * time_sec
    circle_y += speed_y * time_sec

    # Detecção de colisões
    if circle_x <= bar1_x + 10. and bar1_y - 7.5 <= circle_y <= bar1_y + 42.5:
        circle_x = bar1_x + 10  # Reposicionar fora da barra
        speed_x = -speed_x
    elif circle_x >= bar2_x - 15. and bar2_y - 7.5 <= circle_y <= bar2_y + 42.5:
        circle_x = bar2_x - 15  # Reposicionar fora da barra
        speed_x = -speed_x
    elif circle_y <= 10.:
        circle_y = 10.  # Reposicionar no limite superior
        speed_y = -speed_y
    elif circle_y >= 457.5:
        circle_y = 457.5  # Reposicionar no limite inferior
        speed_y = -speed_y

    # Verificar pontuação
    if circle_x < 0:  # Player 2 pontua
        bar2_score += 1
        circle_x, circle_y = 307.5, 232.5  # Resetar posição
        speed_x, speed_y = 150., 150.  # Resetar velocidade
    elif circle_x > 640:  # Player 1 pontua
        bar1_score += 1
        circle_x, circle_y = 307.5, 232.5  # Resetar posição
        speed_x, speed_y = -150., -150.  # Resetar velocidade

    # Atualizar a tela
    pygame.display.update()
    return True


# Loop principal
while True:
    if not update_game():
        break

pygame.quit()
