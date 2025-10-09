# -*- coding: utf-8 -*-
"""
Created on Fri Mar  3 18:00:00 2023

"""

import numpy as np
import matplotlib.pyplot as plt
from matplotlib.pyplot import Circle
from scipy.integrate import solve_ivp

def f1(v_x):
    # The v_x is the velocity value on x-axis
    return v_x

def f2(v_y):
    # The v_y is the velocity value on y-axis
    return v_y

def f3(r_x,r_y,m_p):
    return -G * m_p * r_x / ((r_x**2 + r_y**2)**(3/2))

def f4(r_x,r_y,m_p):
    return -G * m_p * r_y / ((r_x**2 + r_y**2)**(3/2))

# For x-axis
def f13(t,state,r_y,m_p):
    r_x,v_x = state
    dxdt = f1(v_x)
    dv_xdt = f3(r_x,r_y,m_p)
    return (dxdt,dv_xdt)

# For y-axis
def f24(t,state,r_x,m_p):
    r_y,v_y = state
    dydt = f2(v_y)
    dv_ydt = f4(r_x, r_y, m_p)
    return (dydt,dv_ydt)

# Gravitational constant, G, unit: m^3 kg^-1 s^-2
G = 6.6743e-11
# The mass and radius of Earth
m_earth = 5.976e24 # unit: kg
r_earth = 6371e3 # unit: m
# The mass and radius of Moon
m_moon = 0.0123* m_earth
r_moon =  0.272 * r_earth
# Mean distance between Earth and moon
R_moon = 382.5e6 # unit: m

t0 = 0 # Starting time, unit: seconds
tmax = 0 # Finish time (need to update later), unit: seconds
numpoints = 0 # Number of data point (need to update later)
# relative tolerance of solve_ivp method
rtol=1e-5
# (m) absolute tolerance of solve_ivp method
atol=1e-6

user_choice = "o"
while user_choice != "q":
    user_choice = input("Please choose the part: 'a', 'b' or 'q' to quit.")
    
    if user_choice == "a":
        print("You have choose part (a).")
        
        # Creat a dictionary for the mass and radius of the planet
        mass_dict = {"MERCURY":0.0553 * m_earth,
                     "VENUS":0.8150 * m_earth,
                     "EARTH":m_earth,
                     "MOON":m_moon,
                     "MARS":0.1074 * m_earth,
                     "JUPITER":317.89 * m_earth,
                     "SATURN":95.18 * m_earth,
                     "URANUS":14.50 * m_earth,
                     "NEPTUNE":17.24 * m_earth,
                     "PLUTO":0.0025 * m_earth}
        radius_dict = {"MERCURY":0.382 * r_earth,
                     "VENUS":0.949 * r_earth,
                     "EARTH":r_earth,
                     "MOON":r_moon,
                     "MARS":0.532 * r_earth,
                     "JUPITER":11.19 * r_earth,
                     "SATURN":9.41 * r_earth,
                     "URANUS":3.98 * r_earth,
                     "NEPTUNE":3.81 * r_earth,
                     "PLUTO":0.23 * r_earth}
        # Let the user to choose a planet.
        user_planet = input("Choose a planet you want to use:")
        # Turn the string type by user into capital letter.
        user_planet = user_planet.upper()
        m_p = mass_dict[user_planet]
        r_p = radius_dict[user_planet]
        # Let user write the starting position.
        print("Please write ',' between the two values, like 1,4.")
        user_r_x, user_r_y = input(f"Starting position (higher than {r_p:.3e}):").split(",")
        r_x = float(user_r_x)
        r_y = float(user_r_y)
        
        user_vx, user_vy = input("Starting velocity:").split(",")
        v_x = float(user_vx)
        v_y = float(user_vy)
        
        user_tmax, user_numpoints = input("Maximum time, number of data point =").split(",")
        tmax = int(user_tmax)
        numpoints = int(user_numpoints)
        t = np.linspace(t0, tmax,num=numpoints) # Time interval
        m_r = 1 # mass of rocket, unit:kg
        
        # Let the user to choose which method to use.
        user_method_choice = input("Type 'a' for SciPy method, type 'b' for explicit method.")

        if user_method_choice == 'a':
            
            result_x = solve_ivp(f13, (t0,tmax),(r_x,v_x),
                                 args=(r_y,m_p),method='RK45',
                                 t_eval=t,rtol=rtol,atol=atol)
            result_y = solve_ivp(f24, (t0,tmax),(r_y,v_y),
                                 args=(r_x,m_p),method='RK45',
                                 t_eval=t,rtol=rtol,atol=atol)
            # The circle represent the planet.
            circle_planet = Circle((0,0), radius=r_p, fill=False,label=f"{user_planet}")
            # Plot the graph.
            fig, ax = plt.subplots()
            ax.set_xlabel('x-axis (m)')
            ax.set_ylabel('y-axis (m)')
    
            ax.plot(result_x.y[0,:],result_y.y[0,:],label=f"velocity:({user_vx},{user_vy})")
            # Plot the circle as the planet.
            ax.add_patch(circle_planet)
            ax.axis('equal')
            ax.legend()
            # Uncomment the code below to donwload the graph.
            # fig.savefig(f'v_{user_vx}_{user_vy}_p_{user_r_x}_{user_r_y}.png', dpi=900)
            
            # Calculate the total energy
            v = np.sqrt(result_x.y[1,:]**2 + result_y.y[1,:]**2)
            r = np.sqrt(result_x.y[0,:]**2 + result_y.y[0,:]**2)
            E_kinetic = 1/2 * m_r * v**2
            E_grav = -(G * m_p * m_r)/r
            E_total = E_kinetic + E_grav
            
            # The number of times when the total energy is conserved.
            conserved = 0
            not_conserved = 0
            for i in np.arange(numpoints-1, dtype=int):
                if (round(E_total[i])) == (round(E_total[i+1])):
                    conserved += 1
                else:
                    not_conserved += 1
            print(f"The energy is conserved for {conserved} times.")
            print(f"The energy is not conserved for {not_conserved} times.")
            
        
        elif user_method_choice == 'b':
            
            dt = (tmax-t0)/(numpoints-1) # time increment
            r_x_arr = np.zeros(numpoints) # array to hold r_x values
            r_y_arr = np.zeros(numpoints) # array to hold r_y values
            v_x_arr = np.zeros(numpoints) # array to hold v_x values
            v_y_arr = np.zeros(numpoints) # array to hold v_y values
            
            # Assign the initial value for each array.
            r_x_arr[0] = r_x
            r_y_arr[0] = r_y
            
            v_x_arr[0] = v_x
            v_y_arr[0] = v_y
            
            i = 0 # index value
            
            r_value = r_p * 2
            
            while t[i] < tmax and r_value > r_p: # If it hit the planet stop.
                
                # Calculate the r value
                r_value = (r_x_arr[i]**2 + r_y_arr[i]**2)**(1/2)
                
                k1x = f1(v_x_arr[i])
                k1y = f2(v_y_arr[i])
                
                k1vx = f3(r_x_arr[i], r_y_arr[i], m_p)
                k1vy = f4(r_x_arr[i], r_y_arr[i], m_p)
                
                
                k2x = f1(v_x_arr[i] + dt*k1vx/2)
                k2y = f1(v_y_arr[i] + dt*k1vy/2)
                
                k2vx = f3(r_x_arr[i] + dt*k1x/2, r_y_arr[i] + dt*k1y/2, m_p)
                k2vy = f4(r_x_arr[i] + dt*k1x/2, r_y_arr[i] + dt*k1y/2, m_p)
                
                
                k3x = f1(v_x_arr[i] + dt*k2vx/2)
                k3y = f1(v_y_arr[i] + dt*k2vy/2)
                
                k3vx = f3(r_x_arr[i] + dt*k2x/2, r_y_arr[i] + dt*k2y/2, m_p)
                k3vy = f4(r_x_arr[i] + dt*k2x/2, r_y_arr[i] + dt*k2y/2, m_p)
                
                
                k4x = f1(v_x_arr[i] + dt*k3vx)
                k4y = f2(v_y_arr[i] + dt*k3vy)
                
                k4vx = f3(r_x_arr[i] + dt*k3x, r_y_arr[i] + dt*k3y, m_p)
                k4vy = f4(r_x_arr[i] + dt*k3x, r_y_arr[i] + dt*k3y, m_p)
                
                r_x_arr[i+1] = r_x_arr[i] + dt/6*(k1x + 2*k2x + 2*k3x + k4x)
                r_y_arr[i+1] = r_y_arr[i] + dt/6*(k1y + 2*k2y + 2*k3y + k4y)
                
                v_x_arr[i+1] = v_x_arr[i] + dt/6*(k1vx + 2*k2vx + 2*k3vx + k4vx)
                v_y_arr[i+1] = v_y_arr[i] + dt/6*(k1vy + 2*k2vy + 2*k3vy + k4vy)
                
                i += 1
                
            # The circle represent the planet.
            circle_planet = Circle((0,0), radius=r_p, fill=False,label=f"{user_planet}")
            # Plot the graph.
            fig, ax = plt.subplots()
            ax.set_xlabel('x-axis (m)')
            ax.set_ylabel('y-axis (m)')
    
            ax.plot(r_x_arr,r_y_arr,label=f"velocity:({user_vx},{user_vy})")
            # Plot the circle as the planet.
            ax.add_patch(circle_planet)
            ax.axis('equal')
            ax.legend()
            # Uncomment the code below to download the graph.
            # fig.savefig(f'v_{user_vx}_{user_vy}_p_{user_r_x}_{user_r_y}.png', dpi=900)
            
            # Calculate the total energy
            v = np.sqrt(v_x_arr**2 + v_y_arr**2)
            r = np.sqrt(r_x_arr**2 + r_y_arr**2)
            E_kinetic = 1/2 * m_r * v**2
            E_grav = -(G * m_p * m_r)/r
            E_total = E_kinetic + E_grav
            
            # The number of times when the total energy is conserved.
            conserved = 0
            not_conserved = 0
            
            for i in np.arange(numpoints-1, dtype=int):
                if (round(E_total[i])) == (round(E_total[i+1])):
                    conserved += 1
                else:
                    not_conserved += 1
            print(f"The energy is conserved for {conserved} times.")
            print(f"The energy is not conserved for {not_conserved} times.")
                
        
        else:
            print("This is not a valid choice")

    elif user_choice == "b":
        print("You have choose part (b)")
        
# =============================================================================
#             Becasue in this part we have two planet,
#             we need to made some changes to the function, f3 and f4
# =============================================================================
        
        # For x-axis:
        def f3(r_x,r_y):
            return -G * m_earth * r_x / ((r_x**2 + r_y**2)**(3/2)) - G * m_moon * r_x / ((r_x**2 + (r_y - R_moon)**2)**(3/2))
        
        # For y-axis
        def f4(r_x,r_y):
            return -G / ((r_x**2 + r_y**2)**(3/2)) * (r_y * m_earth) - G * m_moon * (r_y - R_moon) / (r_x**2 + (r_y - R_moon)**2)**(3/2)
        # Let the user to type the starting position
        print("Please write ',' between the two values, like 1,4.")
        user_r_x, user_r_y = input(f"Starting position (between {r_earth:.3e} and {R_moon:.3e}):").split(",")
        r_x = float(user_r_x)
        r_y = float(user_r_y)
        
        user_vx, user_vy = input("Starting velocity:").split(",")
        v_x = float(user_vx)
        v_y = float(user_vy)
        
        user_tmax, user_numpoints = input("Maximum time, number of data point =").split(",")
        tmax = int(user_tmax)
        numpoints = int(user_numpoints)
        t = np.linspace(t0, tmax,num=numpoints) # Time interval
        
        dt = (tmax-t0)/(numpoints-1) # time increment
        r_x_arr = np.zeros(numpoints) # array to hold r_x values
        r_y_arr = np.zeros(numpoints) # array to hold r_y values
        v_x_arr = np.zeros(numpoints) # array to hold v_x values
        v_y_arr = np.zeros(numpoints) # array to hold v_y values
        
        # Assign the initial value for each array.
        r_x_arr[0] = r_x
        r_y_arr[0] = r_y
        
        v_x_arr[0] = v_x
        v_y_arr[0] = v_y
        
        i = 0 # index value
        
        r_value = r_earth + 1e6
        
        while t[i] < tmax and r_value > r_earth: # If it hit the planet stop.
            
            # Calculate the r value
            r_value = (r_x_arr[i]**2 + r_y_arr[i]**2)**(1/2)
            
            k1x = f1(v_x_arr[i])
            k1y = f2(v_y_arr[i])
            
            k1vx = f3(r_x_arr[i], r_y_arr[i])
            k1vy = f4(r_x_arr[i], r_y_arr[i])
            
            
            k2x = f1(v_x_arr[i] + dt*k1vx/2)
            k2y = f1(v_y_arr[i] + dt*k1vy/2)
            
            k2vx = f3(r_x_arr[i] + dt*k1x/2, r_y_arr[i] + dt*k1y/2)
            k2vy = f4(r_x_arr[i] + dt*k1x/2, r_y_arr[i] + dt*k1y/2)
            
            
            k3x = f1(v_x_arr[i] + dt*k2vx/2)
            k3y = f1(v_y_arr[i] + dt*k2vy/2)
            
            k3vx = f3(r_x_arr[i] + dt*k2x/2, r_y_arr[i] + dt*k2y/2)
            k3vy = f4(r_x_arr[i] + dt*k2x/2, r_y_arr[i] + dt*k2y/2)
            
            
            k4x = f1(v_x_arr[i] + dt*k3vx)
            k4y = f2(v_y_arr[i] + dt*k3vy)
            
            k4vx = f3(r_x_arr[i] + dt*k3x, r_y_arr[i] + dt*k3y)
            k4vy = f4(r_x_arr[i] + dt*k3x, r_y_arr[i] + dt*k3y)
            
            r_x_arr[i+1] = r_x_arr[i] + dt/6*(k1x + 2*k2x + 2*k3x + k4x)
            r_y_arr[i+1] = r_y_arr[i] + dt/6*(k1y + 2*k2y + 2*k3y + k4y)
            
            v_x_arr[i+1] = v_x_arr[i] + dt/6*(k1vx + 2*k2vx + 2*k3vx + k4vx)
            v_y_arr[i+1] = v_y_arr[i] + dt/6*(k1vy + 2*k2vy + 2*k3vy + k4vy)
            
            i += 1
            
        # The circle represent Earth and moon.
        circle_earth = Circle((0,0), radius=r_earth, fill=False,label="Earth", color='green')
        circle_moon = Circle((0,R_moon), radius=r_moon, label="moon", color='dimgray')
        # Plot the graph.
        fig, ax = plt.subplots()
        ax.set_xlabel('x-axis (m)')
        ax.set_ylabel('y-axis (m)')

        ax.plot(r_x_arr,r_y_arr,label=f"velocity:({user_vx},{user_vy})")
        # Plot the circle as the planet.
        ax.add_patch(circle_earth)
        ax.add_patch(circle_moon)
        ax.axis('equal')
        ax.legend()
        # Uncomment the code below to download the graph.
        # fig.savefig(f'Earth_moon_v_{user_vx}_{user_vy}.png', dpi=900)
        
    elif user_choice != "q":
        print("Please write a valid answer")

print("You had choose to quit the this program.")
print("Thank you for using this program. Goodbye!")