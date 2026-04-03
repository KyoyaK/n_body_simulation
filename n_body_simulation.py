import pygame
import numpy as np
import matplotlib.pyplot as plt



pygame.init()

screen = pygame.display.set_mode((800, 800))
screen.fill((0,0,0))
pygame.display.set_caption("Simulation WIP")

clock = pygame.time.Clock()
timer = 0
font = pygame.font.Font(None, 30)

bodies = 4

collisions = True

total_energy = []
kinetic_energy = []
potential_energy = []

default_r = 8
zoom = 1



# OBJECT INIT

class NBodySimulation:
    def __init__(self, n):
        rng = np.random.default_rng(seed=13)
        self.pos = rng.uniform(200, 600, (n,2))
        self.vel = rng.uniform(-5, 5, (n,2)) 
        self.acc = np.zeros((n, 2))
        self.mass = rng.uniform(1, 2, (n, 1))
        self.radius = np.ones((n, 1)) * default_r
        self.num = len(self.pos)
        self.gravconst = 5000
        self.colors = rng.uniform(0, 255, (n,3))
    
    def pairwise_distances(self):
        pos_dots = np.dot(self.pos, np.transpose(self.pos))
        diagonal_dots = np.diag(pos_dots)
        distances = np.sqrt(diagonal_dots[:,None] + diagonal_dots[None, :] - 2*pos_dots)
        #Fill 0 to avoid dividing by 0
        np.fill_diagonal(distances, 1)
        #nxn
        self.distances = distances
        return distances

    
    def calculate_forces(self, distances):
        #nxn
        d_2 = distances**2
        e0 = 10
        masses_product_matrix = np.dot(self.mass, np.transpose(self.mass))
        vector_pos_differences = self.pos[:, None, :] - self.pos[None,:,:]
        forces_matrix = (masses_product_matrix[:, :, None]*self.gravconst*
                         vector_pos_differences / ((d_2[:, :, None]+e0**2)**1.5))
        #nx2 matrix of total forces acting on i
        total_forces = np.sum(forces_matrix, axis=0)
        delta_a = total_forces / self.mass
        return delta_a
    


    def check_collision(self, distances):
        #nxn matrix of every radius + every other radius
        radii_pairs = self.radius+self.radius.T
        np.fill_diagonal(radii_pairs, 0)
        colliding = distances < (radii_pairs/2)
        each_colliding = np.triu(colliding)
        return each_colliding

        


    def combine_bodies(self, colliding_array):
        colliding_i, colliding_j = np.where(colliding_array)
        i = colliding_i[0]
        j = colliding_j[0]
        comb_vel = ((self.mass[i]*self.vel[i]+
                    self.mass[j]*self.vel[j])
                    /(self.mass[i] + self.mass[j]))
        comb_pos = ((self.mass[i]*self.pos[i]+
                    self.mass[j]*self.pos[j])
                    /(self.mass[i] + self.mass[j]))
        comb_radius = (np.sqrt(self.radius[i]**2+
                       self.radius[j]**2))
        #add rows
        self.pos = np.vstack((self.pos, comb_pos))
        self.vel = np.vstack((self.vel, comb_vel))
        self.mass = np.vstack((self.mass, 
                               self.mass[i] + self.mass[j]))
        self.radius = np.vstack((self.radius, comb_radius))
        #delete rows
        self.mass = np.delete(self.mass, [i, j], axis=0)
        self.vel = np.delete(self.vel, [i, j], axis=0)
        self.pos = np.delete(self.pos, [i, j], axis=0)
        self.radius = np.delete(self.radius, [i, j], axis=0)
        self.num -= 1

        

        
    def update(self, dt, delta_a, method):
        """dt, delta a, and method (Euler or Verlet)"""
        if method == "Euler":
            self.acc = delta_a
            self.vel += delta_a*dt
            self.pos += self.vel*dt
        elif method == "Verlet":
            self.acc = delta_a
            self.pos = self.pos + self.vel*dt + 0.5*self.acc*(dt**2)
            new_acc=self.calculate_forces(self.distances)
            self.vel = self.vel + (new_acc+self.acc)*dt/2
    
    def calculate_energy(self, distances):
        #Kinetic
        speed = np.sqrt(np.sum(self.vel**2, axis=1))
        e_kin = np.sum(0.5*self.mass[:, 0]*(speed**2))
        #Potential
        masses_product_matrix = np.dot(self.mass, np.transpose(self.mass))
        potentials = np.triu(self.gravconst*masses_product_matrix / 
                             np.sqrt(distances**2+100), 1)
        e_pot = np.sum(potentials)
        e_total = e_kin - e_pot
        return e_kin, e_pot, e_total


    def draw(self, screen_cords, screen_radius): 
        for i in range(self.num):
            #placeholder colors
            pygame.draw.circle(screen, self.colors[i], 
                                (screen_cords[i, 0], screen_cords[i, 1]),
                                screen_radius[i,0])
            #velocity line for now
            pygame.draw.line(screen, (255, 255, 255),
                            (screen_cords[i, 0]+self.vel[i, 0]*5, 
                             screen_cords[i, 1]+self.vel[i, 1]*5),
                            (screen_cords[i, 0], screen_cords[i, 1]))
            #stats
            stats = font.render(f"""Body {i+1}
M: {self.mass[i, 0]:.2}
V: {np.sqrt(self.vel[i, 0]**2+self.vel[i, 1]**2):.2}""", 
                            True, (255, 255, 255))
            screen.blit(stats, (screen_cords[i, 0]+10, screen_cords[i, 1]+30))

n_bodies = NBodySimulation(bodies)

#Virial Theorem
initial_d = n_bodies.pairwise_distances()
e_kin_i, e_pot_i, e_total_i = n_bodies.calculate_energy(initial_d)
n_bodies.vel = n_bodies.vel * np.sqrt((e_pot_i/2)/e_kin_i)


#RUNNING THE SIMULATION

running = True
pause = False

screen_cords = n_bodies.pos
screen_radius = n_bodies.radius


while running: 
    for event in pygame.event.get():
        if event.type == pygame.QUIT:
            running = False
        elif event.type == pygame.KEYDOWN:
            if event.key == pygame.K_SPACE:
                pause = not pause
            if event.key == pygame.K_0:
                n_bodies.vel = np.zeros((n_bodies.num, 2))
        elif event.type == pygame.MOUSEBUTTONDOWN:
            if event.button == 1:
                new = np.array([event.pos[0], event.pos[1]])
                #print(bodies)
                n_bodies.pos = np.vstack((n_bodies.pos, new))
                n_bodies.vel = np.vstack((n_bodies.vel, np.zeros((1,2))))
                n_bodies.mass = np.vstack((n_bodies.mass, np.array([1])))
                n_bodies.radius = np.vstack((n_bodies.radius, np.array([default_r])))
                n_bodies.colors = np.vstack((n_bodies.colors, 255*np.ones((1,3))))
                n_bodies.num+=1
    
    key = pygame.key.get_pressed()

    if key[pygame.K_d]:
        n_bodies.pos[:, 0] -= 10
    if key[pygame.K_a]:
        n_bodies.pos[:, 0] += 10
    if key[pygame.K_w]:
        n_bodies.pos[:, 1] += 10
    if key[pygame.K_s]:
        n_bodies.pos[:, 1] -= 10
    if key[pygame.K_UP]:
        zoom +=0.01
    if key[pygame.K_DOWN]:
        zoom -=0.01


    centered_positions = n_bodies.pos-400
    screen_cords = centered_positions*zoom + 400
    screen_radius = zoom*n_bodies.radius

    dt = clock.tick(60)/1000


    
    if not pause:
        #Calculations
        substeps = 20
        for num in range(substeps):
            dist = n_bodies.pairwise_distances()
            delta_a = n_bodies.calculate_forces(dist)
            n_bodies.update(dt, delta_a, "Verlet")
            e_kin, e_pot, e_total = n_bodies.calculate_energy(dist)
            total_energy.append(e_total)
            kinetic_energy.append(e_kin)
            potential_energy.append(e_pot)
            if collisions:
                colliding_array = n_bodies.check_collision(dist)
                if np.any(colliding_array):
                    n_bodies.combine_bodies(colliding_array)

        #Draw
        screen.fill((0, 0, 0))
        n_bodies.draw(screen_cords, screen_radius)
        
        #Timer
        timer += clock.get_time()
        timer_text = font.render(f"{timer/1000:.3} seconds", True, (255, 255, 255))
        screen.blit(timer_text, (50, 50))

        #if timer >= 10000:
        #    running = False

        pygame.display.update()

pygame.quit()

#PLOT ENERGIES OVER TIME
fig, ax = plt.subplots()

e_total_line = ax.plot(total_energy, label="Total Energy")
e_kin_line = ax.plot(kinetic_energy, label="Kinetic Energy", lw=2)
e_pot_line = ax.plot(potential_energy, label="Potential Energy", lw=2)
ax.legend()

plt.show()


    
