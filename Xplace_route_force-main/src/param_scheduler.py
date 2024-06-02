from .database import PlaceData
import torch
import numpy as np
import matplotlib.pyplot as plt
import os
import copy
import math


class MetricRecorder:
    def __init__(self, **kwargs) -> None:
        for key, item in kwargs.items():
            if not isinstance(item, list):
                raise TypeError("%s is not a list for key %s" % (item, key))
            self[key] = item

    def push(self, **kwargs) -> None:
        for key, item in kwargs.items():
            if type(item) == torch.Tensor and item.dim() == 0:
                item = item.item()
            elif np.issubdtype(type(item), np.floating):
                item = float(item)
            elif np.issubdtype(type(item), np.integer):
                item = int(item)
            if not type(item) == int and not type(item) == float:
                raise TypeError(
                    "item %s type(%s) is not a number for key %s"
                    % (item, type(item), key)
                )
            self[key].append(item)

    FORCE_RATIO_KEYS = ["density_force_ratio", "mov_route_force_ratio", "mov_congest_force_ratio", "mov_pseudo_force_ratio", "all_route_force_ratio"]

    def visualize_force_ratio(self, prefix, ax=None):
        if ax is None:
            fig, ax = plt.subplots()

        for key, value in self:
            if key in self.FORCE_RATIO_KEYS:
                x = list(range(len(value)))
                ax.plot(x, value, label=key)

        ax.legend()
        ax.figure.savefig(prefix + "force_ratio.png")
        plt.close(ax.figure)

    def visualize(self, prefix):
        self.visualize_force_ratio(prefix)

        for key, value in self:
            if key in self.FORCE_RATIO_KEYS:
                continue

            fig, ax = plt.subplots()
            x = list(range(len(value)))
            ax.plot(x, value, label=key)
            ax.legend()
            fig.savefig(prefix + f"{key}.png")
            plt.close(fig)

    def __getitem__(self, key):
        return getattr(self, key, None)

    def __setitem__(self, key, value):
        setattr(self, key, value)

    def __delitem__(self, key):
        return delattr(self, key)

    @property
    def keys(self):
        keys = [key for key in self.__dict__.keys() if self[key] is not None]
        return keys

    def __len__(self):
        r"""Returns the number of all present attributes."""
        return len(self.keys)

    def __contains__(self, key):
        r"""Returns :obj:`True`, if the attribute :obj:`key` is present in the
        data."""
        return key in self.keys

    def __iter__(self):
        r"""Iterates over all present attributes in the data, yielding their
        attribute names and content."""
        for key in sorted(self.keys):
            yield key, self[key]


class ParamScheduler:
    def __init__(self, data: PlaceData, args, logger) -> None:
        self.__logger__ = logger
        self.data = data
        self.iter = 0
        self.init_iter = 0
        self.all_init_iters = []
        # metrics
        self.metrics = [
            "hpwl",
            "overflow",
            "mu",
            "wa_coeff",
            "density_weight",
            "precond_coef",
            "weighted_weight",
            "density_force_ratio",# density_force L1-norm / wire_force L1-norm
            "mov_route_force_ratio",
            "mov_congest_force_ratio",
            "mov_pseudo_force_ratio",
            "all_route_force_ratio",
        ]
        self.recorder = MetricRecorder(**{m: [] for m in self.metrics})
        # best solution
        # main solution
        self.best_sol: torch.Tensor = None
        self.best_metric = {"overflow": float("inf"), "hpwl": float("inf")}
        # aux solution
        self.best_sol_aux: torch.Tensor = None
        self.best_metric_aux = {"overflow": float("inf"), "hpwl": float("inf")}
        # rollback solution
        self.best_sol_rollback: torch.Tensor = None
        self.best_metric_rollback = {"overflow": float("inf"), "hpwl": float("inf")}

        # params
        self.precond_coef = 1.0
        self.precond_weight = None
        self.density_weight_start = args.density_weight
        self.density_weight = args.density_weight
        self.density_weight_coef = args.density_weight_coef
        self.wa_coeff = args.wa_coeff
        self.base_gamma = args.wa_coeff * torch.sum(data.unit_len).item()
        self.wa_coeff_start = 10 * self.base_gamma
        self.wa_coeff = 10 * self.base_gamma
        self.use_precond = args.use_precond
        self.mu = 1.0
        self.max_life = 30
        self.life = self.max_life
        self.stop_overflow = args.stop_overflow
        self.skip_update = False if args.enable_skip_update else None
        self.enable_fence = data.enable_fence
        self.min_enlarge_density_interval = 1000
        self.last_enlarge_density_iter = -self.min_enlarge_density_interval
        # skip density force
        self.enable_sample_force = True
        self.density_force_ratio = 0.0

        # routability optimization params
        self.mov_route_force_ratio = 0.0
        self.mov_congest_force_ratio = 0.0
        self.mov_pseudo_force_ratio = 0.0
        self.all_route_force_ratio = 0.0
        self.enable_route = args.use_route_force or args.use_cell_inflate
        self.use_cell_inflate = args.use_cell_inflate
        self.use_route_force = args.use_route_force
        self.use_cell_inflate_and_route_force = args.use_cell_inflate and args.use_route_force
        self.use_cell_inflate_only = args.use_cell_inflate and not args.use_route_force
        
        self.route_weight = args.route_weight
        self.congest_weight = args.congest_weight
        self.base_route_weight = args.route_weight
        self.base_congest_weight = args.congest_weight
        self.pseudo_weight = args.pseudo_weight
        self.num_route_iter = args.num_route_iter
        self.mov_node_to_num_pseudo_pins = None
        self.last_route_iter = None
        self.route_ratio = 1
        self.rerun_route = False
        self.recal_conn_route_force = False
        self.open_cell_inflate_opt = False
        self.open_route_force_opt = False
        self.start_route_iter = None
        self.curr_optimizer_cnt = 0
        self.prev_optimizer_cnt = 0
        self.max_route_opt = 5
        self.max_route_force_opt = 3
        self.wait_router_sol_recorder =[]
        self.gr_sol_recorder = []
        self.grad_recorder = {
            "conn_node_grad_by_wl": None,
            "node_grad_by_density": None,
            "all_route_grad": None,
            "mov_route_grad": None,
            "mov_congest_grad": None,
            "mov_pseudo_grad": None
        }
        # self.gr_sol_metric_recorder = MetricRecorder(**{m: [] for m in self.metrics})
        self.use_norm_density_weight = False
        
    def set_init_param(self, init_density_weight, data: PlaceData, init_density_loss):
        # init_density_weight
        self.init_iter = self.iter
        self.all_init_iters.append(self.init_iter)
        self.precond_coef = 1.0
        if self.use_norm_density_weight:
            self.density_weight = init_density_weight * 1.0
        # elif self.use_cell_inflate and self.curr_optimizer_cnt > 0:
        #     self.density_weight = init_density_weight * 0.1
        else:
            self.density_weight = copy.deepcopy(self.density_weight_start) * init_density_weight
        self.wa_coeff = copy.deepcopy(self.wa_coeff_start)
        self.update_precond_weight(data)

    def set_route_init_param(
        self, init_density_weight, init_route_weight, init_congest_weight, data: PlaceData, args
    ):
        self.init_iter = self.iter
        self.all_init_iters.append(self.init_iter)
        self.precond_coef = 1.0
        
        # init_density_weight
        self.density_weight = init_density_weight * 1.0 #TODO: set rerun density_weight
        
        self.route_weight = init_route_weight * args.route_weight
        self.congest_weight = init_congest_weight * args.congest_weight
        self.pseudo_weight = args.pseudo_weight # same scale as wirelength weight
        # self.wa_coeff = copy.deepcopy(self.wa_coeff_start)
        self.update_precond_weight(data)

    def reset_best_sol(self):
        # best solution
        # main solution
        self.best_sol: torch.Tensor = None
        self.best_metric = {"overflow": float("inf"), "hpwl": float("inf")}
        # aux solution
        self.best_sol_aux: torch.Tensor = None
        self.best_metric_aux = {"overflow": float("inf"), "hpwl": float("inf")}
        # rollback solution
        self.best_sol_rollback: torch.Tensor = None
        self.best_metric_rollback = {"overflow": float("inf"), "hpwl": float("inf")}
        self.life = self.max_life

    def push_metric(self, hpwl, overflow):
        metrics_dict = {
            "hpwl": hpwl,
            "overflow": overflow,
            "mu": self.mu,
            "wa_coeff": self.wa_coeff,
            "density_weight": self.density_weight,
            "precond_coef": self.precond_coef,
            "weighted_weight": self.weighted_weight,
            "density_force_ratio": self.density_force_ratio,
            "mov_route_force_ratio": self.mov_route_force_ratio,
            "mov_congest_force_ratio": self.mov_congest_force_ratio,
            "mov_pseudo_force_ratio": self.mov_pseudo_force_ratio,
            "all_route_force_ratio": self.all_route_force_ratio,
        }
        self.recorder.push(**metrics_dict)

    def push_gr_sol(self, gr_metrics, hpwl, overflow, mov_node_pos: torch.Tensor, can_as_gp_sol: bool =True):
        # can_as_gp_sol: True if the solution can be used as GP solution
        if overflow < 0.15:
            can_as_gp_sol = True
        self.gr_sol_recorder.append((
            gr_metrics, hpwl, overflow, mov_node_pos.detach().clone()
        ))
            
    def reset_gr_sol_recorder(self):
        self.gr_sol_recorder = []
        
    def push_wait_router_sol(self, hpwl, overflow, mov_node_pos: torch.Tensor):
        self.wait_router_sol_recorder.append((hpwl, overflow, mov_node_pos.detach().clone()))
        
    def reset_wait_router_sol_recorder(self):
        self.wait_router_sol_recorder = []

    def step(self, hpwl, overflow, node_pos, data):
        self.update_precond_weight(data)
        self.push_metric(hpwl, overflow)
        self.update_best_sol(node_pos)
        if self.skip_update is not None:
            # if self.density_weight > 0.1 and self.recorder.overflow[-2] < 0.5: # 2021 11 12 best
            #     self.skip_update = np.random.random() > (np.random.randn() * 0.08 + 0.4)
            # if self.density_weight > 0.1 or self.recorder.overflow[-1] < 0.2: # 2021 11 13 best
            #     self.skip_update = np.random.random() > 0.4
            if self.weighted_weight > 0.5 and self.weighted_weight < 0.95:
                self.skip_update = ((self.iter - self.init_iter) % 3 != 0)
            elif self.iter - self.init_iter < 50:
                # slow down the param update of early stage
                self.skip_update = ((self.iter - self.init_iter) % 3 != 0)
            else:
                self.skip_update = False

        if self.use_route_force and self.open_route_force_opt:
            self.step_density_and_route_weight()
        else:
            self.step_density_weight()
        self.step_wa_coeff()
        self.step_precond_coef()
        self.iter += 1

    def step_density_weight(self):
        if self.iter - self.init_iter < 1:
            return
        if self.skip_update is not None:
            if self.skip_update:
                return
        delta_hpwl = self.recorder.hpwl[-1] - self.recorder.hpwl[-2]
        if delta_hpwl < 0:
            self.mu = 1.05 * np.maximum(np.power(0.9999, float(self.iter - self.init_iter)), 0.98)
        else:
            self.mu = 1.05 * np.clip(np.power(1.05, -delta_hpwl / 350000), 0.95, 1.05)
        self.density_weight *= self.mu

        if (
            not self.enable_fence and 
            self.iter > 15 and
            self.iter - self.last_enlarge_density_iter > self.min_enlarge_density_interval and
            self.check_plateau(self.recorder.overflow, window=25, threshold=0.001)
        ):
            if self.recorder.overflow[-1] > 0.9:
                self.last_enlarge_density_iter = self.iter
                self.density_weight *= 2
                self.__logger__.warning(
                    "Detect plateau at early stage, enlarge density_weight. Iter: %d" % 
                self.iter)

    def param_smooth_func(self, input, r=0.2, half_iter=30, end_iter=400):
        logistic = lambda x,k,x_0: 1 / (1 + math.exp(-k * (x - x_0)))
        lhs = 1 - logistic(input, r, end_iter - half_iter)
        rhs = logistic(input, r, half_iter)
        return max(lhs + rhs - 1, 0)

    def step_density_and_route_weight(self):
        self.density_weight = self.calc_norm_density_weight() * 1.0
        self.route_weight = self.calc_norm_route_weight() * self.base_route_weight
        self.congest_weight = self.calc_norm_congest_weight() * self.base_congest_weight
    
    def calc_norm_density_weight(self):
        mov_lhs, mov_rhs = self.data.movable_index
        norm_density_weight = (self.grad_recorder["conn_node_grad_by_wl"][mov_lhs:mov_rhs].norm(p=1) / self.grad_recorder["node_grad_by_density"][mov_lhs:mov_rhs].norm(p=1))
        # non_zero_rows = torch.any(self.grad_recorder["node_grad_by_density"][mov_lhs:mov_rhs] != 0, dim=1).sum().item()
        # density_weight *= non_zero_rows / (mov_rhs - mov_lhs)
        return norm_density_weight
        
    def calc_norm_route_weight(self):
        mov_lhs, mov_rhs = self.data.movable_index
        norm_route_weight = (self.grad_recorder["conn_node_grad_by_wl"][mov_lhs:mov_rhs].norm(p=1) / self.grad_recorder["mov_route_grad"][mov_lhs:mov_rhs].norm(p=1))
        non_zero_rows = torch.any(self.grad_recorder["mov_route_grad"][mov_lhs:mov_rhs] != 0, dim=1).sum().item()
        norm_route_weight *= non_zero_rows / (mov_rhs - mov_lhs)
        return norm_route_weight
    
    def calc_norm_congest_weight(self):
        mov_lhs, mov_rhs = self.data.movable_index
        norm_congest_weight = (self.grad_recorder["conn_node_grad_by_wl"][mov_lhs:mov_rhs].norm(p=1) / self.grad_recorder["mov_congest_grad"][mov_lhs:mov_rhs].norm(p=1))
        non_zero_rows = torch.any(self.grad_recorder["mov_congest_grad"][mov_lhs:mov_rhs] != 0, dim=1).sum().item()
        norm_congest_weight *= non_zero_rows / (mov_rhs - mov_lhs)
        return norm_congest_weight

    def step_wa_coeff(self):
        if self.iter - self.init_iter < 1:
            return
        if self.skip_update is not None:
            if self.skip_update:
                return
        coef = np.power(10, (self.recorder.overflow[-1] - 0.1) * 20 / 9 - 1)
        self.wa_coeff = coef * self.base_gamma

    def step_precond_coef(self):
        if not self.use_precond:
            return
        # if self.use_route_force:
        #     return
        if self.recorder.overflow[self.iter] < 0.3 and self.precond_coef < 1024:
            if (self.iter - self.init_iter) % 20 == 0:
                self.precond_coef *= 2

    def update_precond_weight(self, data: PlaceData):
        if not self.use_precond:
            return
        alpha_1 = data.mov_node_to_num_pins
        alpha_2 = self.precond_coef * self.density_weight * data.mov_node_area
        if self.use_route_force and self.open_route_force_opt:
            #alpha_route = self.route_weight * data.mov_node_to_num_pins
            alpha_congest = self.congest_weight * data.mov_node_area
            #alpha_pseudo = self.pseudo_weight * self.mov_node_to_num_pseudo_pins
            self.precond_weight = (
                alpha_1 + alpha_2  + alpha_congest
                #alpha_1 + alpha_2 + alpha_route + alpha_congest + alpha_pseudo
            ).clamp_(min=1.0)
        else:
            self.precond_weight = (
                alpha_1 + alpha_2
            ).clamp_(min=1.0)
        a2_norm = alpha_2.norm(p=1)
        self.weighted_weight = a2_norm / (alpha_1.norm(p=1) + a2_norm)

    def update_best_sol(self, sol: torch.Tensor) -> None:
        update_flag = False
        hpwl, overflow = self.recorder.hpwl[-1], self.recorder.overflow[-1]
        if self.iter - self.init_iter < 50:
            return update_flag

        if overflow < self.stop_overflow:
            self.life -= 1
            if self.life == self.max_life - 1:
                # release memory of rollback solution
                self.best_sol_rollback = None
                self.best_metric_rollback = {
                    "overflow": float("inf"),
                    "hpwl": float("inf"),
                }
                torch.cuda.empty_cache()

        if (
            overflow < self.stop_overflow * 5
            and overflow >= self.stop_overflow
            and self.life == self.max_life
        ):
            # if overflow < self.best_metric["overflow"]:
            #     if self.best_sol is None:
            #         self.best_sol = sol.detach().clone()
            #     else:
            #         self.best_sol.data.copy_(sol.data)
            #     self.best_metric["hpwl"] = hpwl
            #     self.best_metric["overflow"] = overflow
            if (
                hpwl < self.best_metric_rollback["hpwl"] * 1.01
                and overflow < self.best_metric_rollback["overflow"]
            ):
                if self.best_sol_rollback is None:
                    self.best_sol_rollback = sol.detach().clone()
                else:
                    self.best_sol_rollback.data.copy_(sol.data)
                self.best_metric_rollback["hpwl"] = hpwl
                self.best_metric_rollback["overflow"] = overflow
            update_flag = True

        if (
            overflow < self.stop_overflow
            and hpwl < self.best_metric_aux["hpwl"] * 1.005
            and overflow < self.best_metric_aux["overflow"]
        ):
            if self.best_sol_aux is None:
                self.best_sol_aux = sol.detach().clone()
            else:
                self.best_sol_aux.data.copy_(sol.data)
            self.best_metric_aux["hpwl"] = hpwl
            self.best_metric_aux["overflow"] = overflow
            update_flag = True

        if overflow < self.stop_overflow and hpwl < self.best_metric["hpwl"]:
            if self.best_sol is None:
                self.best_sol = sol.detach().clone()
            else:
                self.best_sol.data.copy_(sol.data)
            self.best_metric["hpwl"] = hpwl
            self.best_metric["overflow"] = overflow
            update_flag = True

        return update_flag

    def need_to_early_stop(self):
        if self.iter - self.init_iter < 100:
            return False
        ptr = self.iter - 1
        if not self.enable_fence and self.check_divergence(
            window=3, threshold=0.01 * self.recorder.overflow[ptr]
        ):
            # dead earlier
            self.life -= 6
        if (
            self.recorder.overflow[ptr] < self.stop_overflow * 5
            and self.recorder.overflow[ptr] >= self.stop_overflow
        ):
            if self.check_plateau(self.recorder.overflow, window=50, threshold=0.05):
                # kill the program since it has converged
                self.__logger__.warning(
                    "Large plateau detected. Kill the optimization process."
                )
                self.life -= self.max_life
        if self.life <= 0:
            return True
        # if (
        #     self.recorder.overflow[ptr] < self.stop_overflow
        #     and self.recorder.hpwl[ptr] > self.recorder.hpwl[ptr - 1]
        # ):
        #     return True
        if (
            self.recorder.overflow[ptr] > self.recorder.overflow[ptr - 1]
            and self.recorder.hpwl[ptr] > self.best_metric["hpwl"] * 2
        ):
            return True
        return False

    def check_plateau(self, x, window=10, threshold=0.001):
        if len(x) < window:
            return False
        x = x[-window:]
        return (np.max(x) - np.min(x)) / np.mean(x) < threshold

    def check_divergence(self, window=50, threshold=0.05, window2=30):
        logger = self.__logger__
        if self.recorder.density_force_ratio[-1] >= 15 and self.iter - self.init_iter >= window2:
            density_force_window_mean = np.mean(self.recorder.density_force_ratio[-window2:]).item()
            if density_force_window_mean >= 15:
                overflow = np.array(self.recorder.overflow[-window2:], dtype=np.float32)
                if overflow[-1] - overflow[0] > 0.001:
                    logger.warning(
                        f"Divergence detected:density force ratio is large ({density_force_window_mean:.2f} > 10) but overflow increases ({overflow[-1] - overflow[0]:.4f})")
                    return True
        
        if self.best_metric["hpwl"] == float("inf"):
            return False
        if self.iter - self.init_iter <= window:
            return False
        x = np.array(self.recorder.hpwl[-window:], dtype=np.float32)
        wl_mean = np.mean(x).item()
        wl_ratio = (wl_mean - self.best_metric["hpwl"]) / self.best_metric["hpwl"]
        if wl_ratio > threshold * 1.2:
            y = np.array(self.recorder.overflow[-window:], dtype=np.float32)
            overflow_mean = np.mean(y).item()
            overflow_diff = np.sum(np.maximum(0, np.sign(y[1:] - y[:-1]))) / len(y[1:])
            overflow_range = np.max(y) - np.min(y)
            overflow_ratio = (
                overflow_mean - max(self.stop_overflow, self.best_metric["overflow"])
            ) / self.best_metric["overflow"]
            if overflow_ratio > threshold:
                logger.warning(
                    f"Divergence detected: overflow increases too much than best overflow ({overflow_ratio:.4f} > {threshold:.4f})"
                )
                return True
            elif overflow_range / overflow_mean < threshold:
                logger.warning(
                    f"Divergence detected: overflow plateau ({overflow_range/overflow_mean:.4f} < {threshold:.4f})"
                )
                return True
            elif overflow_diff > 0.6:
                logger.warning(
                    f"Divergence detected: overflow fluctuate too frequently ({overflow_diff:.2f} > 0.6)"
                )
                return True
            else:
                return False
        else:
            return False

    def get_best_solution(self):
        best_sol = None
        best_hpwl = None
        best_overflow = None
        solution_type = 0
        logger = self.__logger__
        if self.best_sol_rollback is not None:
            best_sol = self.best_sol_rollback.data
            best_hpwl = self.best_metric_rollback["hpwl"]
            best_overflow = self.best_metric_rollback["overflow"]
            solution_type = 3
        elif self.best_sol is None and self.best_sol_aux is None:
            solution_type = 0
        elif self.best_sol_aux is None:
            best_sol = self.best_sol.data
            best_hpwl = self.best_metric["hpwl"]
            best_overflow = self.best_metric["overflow"]
            solution_type = 1
        elif self.best_sol is None:
            best_sol = self.best_sol_aux.data
            best_hpwl = self.best_metric_aux["hpwl"]
            best_overflow = self.best_metric_aux["overflow"]
            solution_type = 2
        else:
            if (
                self.best_metric_aux["hpwl"] < self.best_metric["hpwl"] * 1.005
                and self.best_metric_aux["overflow"] * 1.1
                < self.best_metric["overflow"]
            ):
                best_sol = self.best_sol_aux.data
                best_hpwl = self.best_metric_aux["hpwl"]
                best_overflow = self.best_metric_aux["overflow"]
                solution_type = 2
            else:
                best_sol = self.best_sol.data
                best_hpwl = self.best_metric["hpwl"]
                best_overflow = self.best_metric["overflow"]
                solution_type = 1

        if solution_type == 0:
            logger.info("Cannot find best solution. Use the last solution.")
        elif solution_type == 1:
            logger.info(
                "Find best solution (type %d HPWL driven) masked_hpwl: %.4E overflow: %.4f"
                % (solution_type, best_hpwl, best_overflow)
            )
        elif solution_type == 2:
            logger.info(
                "Find best solution (type %d OVFL driven) masked_hpwl: %.4E overflow: %.4f"
                % (solution_type, best_hpwl, best_overflow)
            )
        elif solution_type == 3:
            logger.info(
                "Cannot find best solution. Use roll back solution (type %d) masked_hpwl: %.4E overflow: %.4f"
                % (solution_type, best_hpwl, best_overflow)
            )
        else:
            raise NotImplementedError("Unknown solution type")

        return best_sol, best_hpwl, best_overflow

    def get_best_gr_sol(self):
        weight = [0.5, 4, 500] # cugr setting, WL, Vias, Shorts
        best_idx = -1
        best_value = float('inf')
        best_sol = None
        best_gr_metrics = None
        for idx, (gr_metrics, hpwl, overflow, mov_node_pos) in enumerate(self.gr_sol_recorder):
            numOvflNets, gr_wirelength, gr_numVias, gr_numShorts, rc_hor_mean, rc_ver_mean = gr_metrics
            # gr_score = weight[0] * gr_wirelength + weight[1] * gr_numVias + weight[1] * gr_numShorts
            if gr_numShorts < best_value:
                # NOTE: I think gr_numShorts is the most important metric...
                best_value = gr_numShorts
                best_idx = idx    

        best_sol = self.gr_sol_recorder[best_idx][3].data
        best_sol_hpwl = self.gr_sol_recorder[best_idx][1]    
        best_sol_overflow = self.gr_sol_recorder[best_idx][2]    
        best_gr_metrics = self.gr_sol_recorder[best_idx][0]
        logger = self.__logger__
        numOvflNets, gr_wirelength, gr_numVias, gr_numShorts, rc_hor_mean, rc_ver_mean = best_gr_metrics
        num_sols = len(self.gr_sol_recorder)
        logger.info(
            "Select best GR solution in routability iteration %d/%d: #OvflNets: %d, "
            "GR WL: %d, GR #Vias: %d, #EstShorts: %d, RC Hor: %.3f, RC Ver: %.3f, "
            "HPWL: %.3E, OVFL: %.4f"%
            (best_idx, num_sols-1, numOvflNets, gr_wirelength, gr_numVias, gr_numShorts, rc_hor_mean, rc_ver_mean, best_sol_hpwl, best_sol_overflow)
        )
        return best_sol

    def need_to_stop_route_force_opt(self):
        if self.start_route_iter is not None and self.iter - self.start_route_iter > self.num_route_iter:
            self.__logger__.info("End route optimization : iteration limit")
            return True
        elif self.check_routability_opt_plateau(window=5, threshold=0.01):
            self.__logger__.info("End route optimization : routability plateau")
            return True
        elif self.check_routability_opt_divergence(window=3, threshold=0.005):
            self.__logger__.info("End route optimization : routability divergence")
            return True
        return False
    
    def check_routability_opt_plateau(self, window=3, threshold=0.01):
        gr_numShorts = [gr_metrics[3] for gr_metrics, hpwl, overflow, mov_node_pos in self.gr_sol_recorder]
        return self.check_plateau(gr_numShorts, window, threshold)
        

    def check_routability_opt_divergence(self, window=3, threshold=0.005):
        if len(self.gr_sol_recorder) < window:
            return False

        gr_numShorts = [gr_metrics[3] for gr_metrics, hpwl, overflow, mov_node_pos in self.gr_sol_recorder[-window:]]

        increase_times = 0
        for i in range(1, len(gr_numShorts)):
            is_increase = gr_numShorts[i] - gr_numShorts[i-1] > threshold * gr_numShorts[i-1]
            increase_times += is_increase

        if increase_times > 1:
            return True
        return False

    def visualize(self, args, logger):
        file_prefix = "%s_" % args.design_name
        res_root = os.path.join(args.result_dir, args.exp_id)
        prefix = os.path.join(res_root, args.eval_dir, file_prefix)
        if not os.path.exists(os.path.dirname(prefix)):
            os.makedirs(os.path.dirname(prefix))
        self.recorder.visualize(prefix)