import numpy as np
import gymnasium as gym
from gymnasium import spaces
from typing import Callable, Optional
from scipy.sparse import diags, csr_matrix
from scipy.sparse.linalg import spsolve
from pde_control_gym.src.environments1d.base_env_1d import PDEEnv1D
from scipy.sparse.linalg import spsolve
from scipy.sparse import csc_matrix

class ReactionDiffusionPDE1D(PDEEnv1D):
    r""" 
    Reaction-Diffusion PDE 1D

    This class implements the 1D Reaction-Diffusion PDE and inhertis from the class :class:`PDEEnv1D`. Thus, for a full list of of arguments, first see the class :class:`PDEEnv1D` in conjunction with the arguments presented here

    :param sensing_noise_func: Takes in a function that can add sensing noise into the system. Must return the same sensing vector as given as a parameter.
    :param reset_init_condition_func: Takes in a function used during the reset method for setting the initial PDE condition :math:`u(x, 0)`.
    :param reset_recirculation_func: Takes in a function used during the reset method for setting the initial plant parameter :math:`\beta` vector at the start of each epsiode.
    :param sensing_loc: Sets the sensing location as either ``"full"``, ``"collocated"``, or ``"opposite"`` which indicates whether the full state, the boundary at the same side of the control, or boundary at the opposite side of control is given as the observation at each time step.
    :param control_type: The control location can either be given as a ``"Dirchilet"`` or ``"Neumann"`` boundary conditions and is always at the ``X`` point. 
    :param sensing_type: Only used when ``sensing_loc`` is set to ``opposite``. In this case, the sensing can be either given as ``"Dirchilet"`` or ``"Neumann"`` and is given at the ``0`` point.
    :param limit_pde_state_size: This is a boolean which will terminate the episode early if :math:`\|u(x, t)\|_{L_2} \geq` ``max_state_value``.
    :param max_state_value: Only used when ``limit_pde_state_size`` is ``True``. Then, this sets the value for which the :math:`L_2` norm of the PDE will be compared to at each step asin ``limit_pde_state_size``.
    :param max_control_value: Sets the maximum control value input as between [``-max_control_value``, ``max_control_value``] and is used in the normalization of action inputs.
    :param control_sample_rate: Sets the sample rate at which the controller is applied to the PDE. This allows the PDE to be simulated at a smaller resolution then the controller.
    """
    def __init__(self, sensing_noise_func: Callable[[np.ndarray], np.ndarray],
                 reset_init_condition_func: Callable[[int], np.ndarray],
                 reset_recirculation_func: Callable[[int], np.ndarray], 
                 sensing_loc: str = "full",
                 control_type: str= "Dirchilet", 
                 sensing_type: str = "Dirchilet", 
                 limit_pde_state_size: bool = False, 
                 max_state_value: float = 1e10, 
                 max_control_value: float = 20, 
                 control_sample_rate: float=0.01,
                 **kwargs):
        super().__init__(**kwargs)
        self.sensing_noise_func = sensing_noise_func
        self.reset_init_condition_func = reset_init_condition_func 
        self.reset_recirculation_func = reset_recirculation_func
        self.sensing_loc = sensing_loc
        self.control_type = control_type
        self.sensing_type = sensing_type
        self.limit_pde_state_size = limit_pde_state_size
        self.max_state_value = max_state_value
        self.max_control_value = max_control_value
        self.control_sample_rate = control_sample_rate
        # Observation space changes depending on sensing
        match self.sensing_loc:
            case "full":
                self.observation_space = spaces.Box(
                    np.full(2*(self.nx+1), -self.max_state_value, dtype="float32"),
                    np.full(2*(self.nx+1), self.max_state_value, dtype="float32"),
                )
            case "collocated" | "opposite":
                self.observation_space = spaces.Box(
                    np.full(1, -self.max_state_value, dtype="float32"),
                    np.full(1, self.max_state_value, dtype="float32"),
                )
            case _:
                raise Exception(
                    "Invalid sensing_loc parameter. Please use 'full', 'collocated', or 'opposite'. See documentation for details."
                )

        # Setup configurations for control and sensing. Messy, but done once, explicitly before runtime to setup return and control functions
        # There is a trick here where noise is a function call itself. Important that noise is a single argument function that returns a single argument
        match self.control_type:
            case "Neumann":
                self.control_update = lambda control, state, dx: control * dx + state
                match self.sensing_loc:
                    # Neumann control u_x(1), full state measurement
                    case "full":
                        self.sensing_update = lambda state, dx, noise: noise(state)
                    # Neumann control u_x(1), Dirchilet sensing u(1)
                    case "collocated":
                        self.sensing_update = lambda state, dx, noise: noise(state[-1])
                    case "opposite":
                        match self.sensing_type:
                            # Neumann control u_x(1), Neumann sensing u_x(0)
                            case "Neumann":
                                self.sensing_update = lambda state, dx, noise: noise(
                                    (state[1] - state[0]) / dx
                                )
                            # Neumann control u_x(1), Dirchilet sensing u(0)
                            case "Dirchilet":
                                raise Exception("In the parabolic PDE system, u(0, t)=0 and so Dirchilet sensing at u(0, t) is not viable. See documentation for details.")
                            case _:
                                raise Exception(
                                    "Invalid sensing_type parameter. Please use 'Neumann' or 'Dirchilet'. See documentation for details."
                                )
                    case _:
                        raise Exception(
                            "Invalid sensing_loc parameter. Please use 'full', 'collocated', or 'opposite'. See documentation for details."
                        )
            case "Dirchilet":
                self.control_update = lambda control, state, dt: control
                match self.sensing_loc:
                    # Dichilet control u(1), full state measurement
                    case "full":
                        self.sensing_update = lambda state, dx, noise: noise(state)
                    # Dichilet control u(1), Neumann sensing u_x(1): Please pass both an initial condition and a recirculation function in the parameters dictionary. See documentation for more details
                    case "collocated":
                        self.sensing_update = lambda state, dx, noise: noise(
                            (state[-1] - state[-2]) / dx
                        )
                    case "opposite":
                        match self.sensing_type:
                            # Dichilet control u(1), Neumann sensing u_x(0)
                            case "Neumann":
                                self.sensing_update = lambda state, dx, noise: noise(
                                    (state[1] - state[0]) / dx
                                )
                            # Dirchilet control u(1), Dirchilet sensing u(0)
                            case "Dirchilet":
                                raise Exception("In the parabolic PDE system, u(0, t)=0 and so Dirchilet sensing at u(0, t) is not viable. See documentation for details.")
                            case _:
                                raise Exception(
                                    "Invalid sensing_type parameter. Please use 'Neumann' or 'Dirchilet'. See documentation for details."
                                )
            case _:
                raise Exception(
                    "Invalid control_type parameter. Please use 'Neumann' or 'Dirchilet'. See documentation for details."
                )

    
    def _build_implicit_matrix(self, dt, dx, beta, control_type):
        """
        Build the coefficient matrix for the implicit finite difference scheme.
        
        For the reaction-diffusion equation:
        (1 + 2F - dt*β)u_i^{n+1} - F*u_{i-1}^{n+1} - F*u_{i+1}^{n+1} = u_i^n
        
        where F = dt/dx²
        """
        F = dt / (dx**2)
        N = self.Nx - 1  # Interior points
        
        # Main diagonal
        main_diag = np.ones(N) * (1 + 2 * F - dt * beta[1:N+1])
        # Off diagonals
        lower_diag = np.ones(N-1) * (-F)
        upper_diag = np.ones(N-1) * (-F)
        
        # Create sparse matrix
        diagonals = [main_diag, lower_diag, upper_diag]
        offsets = [0, -1, 1]
        A = diags(diagonals, offsets, shape=(N, N), format='csr')
        
        return A
        
    def step(self, control):
        """
        step
        
        Moves the PDE with control action forward using an implicit method.
        
        :param control: The control input to apply to the PDE at the boundary.
        """
        dx = self.dx
        dt = self.dt
        self.Nx = self.nx
        sample_rate = int(round(self.control_sample_rate/dt))
        i = 0
        
        while i < sample_rate and self.time_index < self.nt-1:
            self.time_index += 1
            current_time = self.time_index
            
            # Build the implicit matrix for this time step
            self.A = self._build_implicit_matrix(dt, dx, self.beta, self.control_type)
            
            # Set up the right-hand side vector (RHS)
            rhs = self.u[current_time-1, 1:self.Nx].copy()  # Interior points
            
            # Apply boundary conditions to the RHS
            if self.control_type == "Dirchilet":
                # Dirichlet control u(1, t) = control
                rhs[-1] += (dt / dx**2) * self.normalize(control, self.max_control_value)
            else:  # Neumann control
                # Neumann control u_x(1, t) = control
                # Using second-order one-sided difference for the Neumann condition
                rhs[-1] += (dt / dx**2) * (2 * dx * self.normalize(control, self.max_control_value) + self.u[current_time-1, self.Nx-1])
            
            # Solve the linear system A * u^{n+1} = rhs
            u_interior = spsolve(self.A, rhs)
            
            # Update the solution array
            self.u[current_time, 1:self.Nx] = u_interior
            self.u[current_time, 0] = 0  # Dirichlet BC at x=0
            
            # Apply control at the boundary x=1
            if self.control_type == "Dirchilet":
                self.u[current_time, -1] = self.normalize(control, self.max_control_value)
            else:  # Neumann
                # Update ghost point to enforce Neumann condition
                self.u[current_time, -1] = self.u[current_time, -2] + dx * self.normalize(control, self.max_control_value)
            
            i += 1
        
        terminate = self.terminate()
        truncate = self.truncate()
       
        return (
            np.concatenate([self.beta,self.sensing_update(self.u[self.time_index],self.dx,self.sensing_noise_func,)]).astype(np.float32),
            #self.sensing_update(self.u[self.time_index],self.dx,self.sensing_noise_func,).astype(np.float32),
            self.reward_class.reward(self.u, self.time_index, terminate, truncate, self.u[self.time_index][-1]),
            terminate,
            truncate, 
            {},
        )
        
    def terminate(self):
        """
        terminate

        Determines whether the episode should end if the ``T`` timesteps are reached
        """
        if self.time_index >= self.nt - 1:
            return True
        else:
            return False

    def truncate(self):
        """
        truncate 

        Determines whether to truncate the episode based on the PDE state size and the vairable ``limit_pde_state_size`` given in the PDE environment intialization.
        """
        if (
            self.limit_pde_state_size
            and np.linalg.norm(self.u[self.time_index], 2)  >= self.max_state_value
        ):
            return True
        else:
            return False

    # Resets the system state
    def reset(self, seed: Optional[int]=None, options: Optional[dict]=None):
        """
        reset 

        :param seed: Allows a seed for initialization of the envioronment to be set for RL algorithms.
        :param options: Allows a set of options for the initialization of the environment to be set for RL algorithms.

        Resets the PDE at the start of each environment according to the parameters given during the PDE environment intialization
        """
        try:
            init_condition = self.reset_init_condition_func(self.nx)
            beta = self.reset_recirculation_func(self.nx, self.lamArr)
        except:
            raise Exception(
                "Please pass both an initial condition and a recirculation function in the parameters dictionary. See documentation for more details"
                )
        self.u = np.zeros(
            (self.nt, self.nx+1), dtype=np.float32
        )
        self.beta = beta
        
        self.u[0] = init_condition
        self.time_index = 0
        
        return (
            
            np.concatenate([self.beta,self.sensing_update(self.u[self.time_index],self.dx,self.sensing_noise_func,)]).astype(np.float32),
            #self.sensing_update(self.u[self.time_index],self.dx,self.sensing_noise_func,).astype(np.float32),
            {},
        )