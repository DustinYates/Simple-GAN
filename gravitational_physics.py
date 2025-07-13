import numpy as np
from typing import List, Tuple, Dict
from dataclasses import dataclass
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D

@dataclass
class Planet:
    """Represents a planet with mass, position, and velocity"""
    name: str
    mass: float  # in kg
    position: np.ndarray  # 3D position vector [x, y, z] in meters
    velocity: np.ndarray  # 3D velocity vector [vx, vy, vz] in m/s
    radius: float = 0.0  # in meters (for visualization)
    
    def __post_init__(self):
        """Ensure position and velocity are numpy arrays"""
        self.position = np.array(self.position, dtype=float)
        self.velocity = np.array(self.velocity, dtype=float)

class GravitationalSystem:
    """Handles gravitational calculations for multiple planets"""
    
    def __init__(self, planets: List[Planet] = None):
        self.planets = planets or []
        self.G = 6.67430e-11  # Gravitational constant in m³/(kg·s²)
        self.time_step = 1.0  # Time step for simulation in seconds
        
    def add_planet(self, planet: Planet):
        """Add a planet to the system"""
        self.planets.append(planet)
    
    def calculate_gravitational_force(self, planet1: Planet, planet2: Planet) -> np.ndarray:
        """
        Calculate gravitational force between two planets using Newton's law of gravitation
        
        F = G * (m1 * m2) / r² * r̂
        
        Args:
            planet1: First planet
            planet2: Second planet
            
        Returns:
            Force vector from planet1 to planet2
        """
        # Calculate displacement vector from planet1 to planet2
        displacement = planet2.position - planet1.position
        distance = np.linalg.norm(displacement)
        
        # Avoid division by zero if planets are at same position
        if distance < 1e-10:
            return np.zeros(3)
        
        # Calculate unit vector in direction of displacement
        unit_vector = displacement / distance
        
        # Calculate gravitational force magnitude
        force_magnitude = self.G * planet1.mass * planet2.mass / (distance ** 2)
        
        # Return force vector
        return force_magnitude * unit_vector
    
    def calculate_total_force_on_planet(self, target_planet: Planet) -> np.ndarray:
        """
        Calculate total gravitational force on a planet from all other planets
        
        Args:
            target_planet: Planet to calculate forces on
            
        Returns:
            Total force vector on the target planet
        """
        total_force = np.zeros(3)
        
        for planet in self.planets:
            if planet != target_planet:
                force = self.calculate_gravitational_force(planet, target_planet)
                total_force += force
        
        return total_force
    
    def calculate_acceleration(self, planet: Planet) -> np.ndarray:
        """
        Calculate acceleration of a planet using F = ma
        
        Args:
            planet: Planet to calculate acceleration for
            
        Returns:
            Acceleration vector
        """
        total_force = self.calculate_total_force_on_planet(planet)
        return total_force / planet.mass
    
    def update_planet_positions(self, dt: float = None):
        """
        Update positions and velocities of all planets using Euler integration
        
        Args:
            dt: Time step (uses self.time_step if None)
        """
        if dt is None:
            dt = self.time_step
            
        for planet in self.planets:
            # Calculate acceleration
            acceleration = self.calculate_acceleration(planet)
            
            # Update velocity: v = v + a*dt
            planet.velocity += acceleration * dt
            
            # Update position: x = x + v*dt
            planet.position += planet.velocity * dt
    
    def simulate_system(self, total_time: float, time_step: float = None) -> Dict[str, List[np.ndarray]]:
        """
        Simulate the gravitational system over time
        
        Args:
            total_time: Total simulation time in seconds
            time_step: Time step for simulation (uses self.time_step if None)
            
        Returns:
            Dictionary with planet names as keys and lists of positions as values
        """
        if time_step is None:
            time_step = self.time_step
            
        num_steps = int(total_time / time_step)
        trajectories = {planet.name: [] for planet in self.planets}
        
        # Store initial positions
        for planet in self.planets:
            trajectories[planet.name].append(planet.position.copy())
        
        # Simulate over time
        for step in range(num_steps):
            self.update_planet_positions(time_step)
            
            # Store positions
            for planet in self.planets:
                trajectories[planet.name].append(planet.position.copy())
        
        return trajectories
    
    def calculate_orbital_period(self, planet1: Planet, planet2: Planet) -> float:
        """
        Calculate orbital period for a two-body system using Kepler's Third Law
        
        T² = (4π² * a³) / (G * (m1 + m2))
        
        Args:
            planet1: First planet
            planet2: Second planet
            
        Returns:
            Orbital period in seconds
        """
        distance = np.linalg.norm(planet2.position - planet1.position)
        total_mass = planet1.mass + planet2.mass
        
        # Semi-major axis approximation (assuming circular orbit)
        semi_major_axis = distance
        
        # Calculate orbital period
        period = np.sqrt((4 * np.pi**2 * semi_major_axis**3) / (self.G * total_mass))
        
        return period
    
    def calculate_escape_velocity(self, planet: Planet) -> float:
        """
        Calculate escape velocity from a planet's surface
        
        v_escape = sqrt(2GM/r)
        
        Args:
            planet: Planet to calculate escape velocity for
            
        Returns:
            Escape velocity in m/s
        """
        if planet.radius <= 0:
            raise ValueError("Planet radius must be positive for escape velocity calculation")
        
        escape_velocity = np.sqrt(2 * self.G * planet.mass / planet.radius)
        return escape_velocity
    
    def get_system_energy(self) -> Dict[str, float]:
        """
        Calculate total kinetic and potential energy of the system
        
        Returns:
            Dictionary with 'kinetic' and 'potential' energy values
        """
        kinetic_energy = 0.0
        potential_energy = 0.0
        
        # Calculate kinetic energy: KE = 0.5 * m * v²
        for planet in self.planets:
            velocity_magnitude = np.linalg.norm(planet.velocity)
            kinetic_energy += 0.5 * planet.mass * velocity_magnitude**2
        
        # Calculate gravitational potential energy: PE = -G * m1 * m2 / r
        for i, planet1 in enumerate(self.planets):
            for j, planet2 in enumerate(self.planets):
                if i < j:  # Avoid double counting
                    distance = np.linalg.norm(planet2.position - planet1.position)
                    if distance > 1e-10:  # Avoid division by zero
                        potential_energy -= self.G * planet1.mass * planet2.mass / distance
        
        return {
            'kinetic': kinetic_energy,
            'potential': potential_energy,
            'total': kinetic_energy + potential_energy
        }
    
    def visualize_system(self, trajectories: Dict[str, List[np.ndarray]] = None, 
                        show_orbits: bool = True, show_planets: bool = True):
        """
        Visualize the planetary system in 3D
        
        Args:
            trajectories: Optional trajectory data from simulation
            show_orbits: Whether to show orbital paths
            show_planets: Whether to show current planet positions
        """
        fig = plt.figure(figsize=(12, 8))
        ax = fig.add_subplot(111, projection='3d')
        
        # Plot trajectories if provided
        if trajectories and show_orbits:
            for planet_name, positions in trajectories.items():
                positions = np.array(positions)
                ax.plot(positions[:, 0], positions[:, 1], positions[:, 2], 
                       label=f'{planet_name} orbit', alpha=0.7)
        
        # Plot current planet positions
        if show_planets:
            for planet in self.planets:
                ax.scatter(planet.position[0], planet.position[1], planet.position[2], 
                          s=100, label=planet.name)
        
        ax.set_xlabel('X (m)')
        ax.set_ylabel('Y (m)')
        ax.set_zlabel('Z (m)')
        ax.set_title('Gravitational System Visualization')
        ax.legend()
        plt.show()

# Example usage and test functions
def create_solar_system_example():
    """Create a simplified solar system with Earth, Moon, and Sun"""
    
    # Sun (massive central body)
    sun = Planet(
        name="Sun",
        mass=1.989e30,  # kg
        position=[0, 0, 0],  # m
        velocity=[0, 0, 0],  # m/s
        radius=6.96e8  # m
    )
    
    # Earth
    earth = Planet(
        name="Earth",
        mass=5.972e24,  # kg
        position=[1.496e11, 0, 0],  # 1 AU from Sun
        velocity=[0, 29780, 0],  # Orbital velocity
        radius=6.371e6  # m
    )
    
    # Moon
    moon = Planet(
        name="Moon",
        mass=7.342e22,  # kg
        position=[1.496e11 + 3.844e8, 0, 0],  # Earth + Moon distance
        velocity=[0, 29780 + 1022, 0],  # Earth velocity + Moon orbital velocity
        radius=1.737e6  # m
    )
    
    return GravitationalSystem([sun, earth, moon])

def test_gravitational_calculations():
    """Test the gravitational physics calculations"""
    
    # Create test planets
    planet1 = Planet("Test1", 1e24, [0, 0, 0], [0, 0, 0])
    planet2 = Planet("Test2", 1e24, [1e8, 0, 0], [0, 0, 0])
    
    system = GravitationalSystem([planet1, planet2])
    
    # Test gravitational force calculation
    force = system.calculate_gravitational_force(planet1, planet2)
    print(f"Gravitational force between planets: {force} N")
    
    # Test acceleration calculation
    acceleration = system.calculate_acceleration(planet2)
    print(f"Acceleration of planet2: {acceleration} m/s²")
    
    # Test orbital period
    period = system.calculate_orbital_period(planet1, planet2)
    print(f"Orbital period: {period/86400:.2f} days")
    
    # Test system energy
    energy = system.get_system_energy()
    print(f"System energy - Kinetic: {energy['kinetic']:.2e} J, "
          f"Potential: {energy['potential']:.2e} J, "
          f"Total: {energy['total']:.2e} J")

if __name__ == "__main__":
    # Run tests
    test_gravitational_calculations()
    
    # Create and simulate solar system
    solar_system = create_solar_system_example()
    
    # Simulate for 1 year (365 days)
    trajectories = solar_system.simulate_system(365 * 24 * 3600, 24 * 3600)  # 1 day time steps
    
    # Visualize the system
    solar_system.visualize_system(trajectories)
    
    # Print system energy
    energy = solar_system.get_system_energy()
    print(f"\nSolar System Energy:")
    print(f"Kinetic Energy: {energy['kinetic']:.2e} J")
    print(f"Potential Energy: {energy['potential']:.2e} J")
    print(f"Total Energy: {energy['total']:.2e} J") 