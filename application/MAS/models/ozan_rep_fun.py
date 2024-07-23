import math
import statistics

import torch.autograd

from .ozan_min_norm_solvers import MinNormSolver


class OzanRepFunction(torch.autograd.Function):
    n = 5

    def __init__(self):
        super(OzanRepFunction, self).__init__()

    @staticmethod
    def forward(ctx, input):

        shape = input.shape
        ret = input.expand(OzanRepFunction.n, *shape)
        return ret.clone()  # REASON FOR ERROR: forgot to .clone() here

    @staticmethod
    def backward(ctx, grad_output):
        num_grads = grad_output.shape[0]
        batch_size = grad_output.shape[1]
        if num_grads >= 2:
            # print ('shape in = ',grad_output[0].view(batch_size,-1).float().shape)
            try:
                alphas, score = MinNormSolver.find_min_norm_element(
                    [grad_output[i].view(batch_size, -1).float() for i in range(num_grads)])
                # print(alphas)
            except ValueError as error:
                alphas = [1 / num_grads for i in range(num_grads)]

            grad_outputs = [grad_output[i] * alphas[i] * math.sqrt(num_grads) for i in range(num_grads)]
            output = grad_outputs[0]
            for i in range(1, num_grads):
                output += grad_outputs[i]
            return output


        elif num_grads == 1:
            grad_input = grad_output.clone()
            out = grad_input.sum(dim=0)
        else:
            pass
        return out


ozan_rep_function = OzanRepFunction.apply


class TrevorRepFunction(torch.autograd.Function):
    n = 5

    def __init__(self):
        super(TrevorRepFunction, self).__init__()

    @staticmethod
    def forward(ctx, input):
        return input.clone()

    @staticmethod
    def backward(ctx, grad_output):
        grad_input = grad_output.clone()
        mul = 1.0 / math.sqrt(TrevorRepFunction.n)
        out = grad_input * mul
        return out


trevor_rep_function = TrevorRepFunction.apply

count = 0


class GradNormRepFunction(torch.autograd.Function):
    n = 5
    inital_task_losses = None
    current_task_losses = None
    current_weights = None

    def __init__(self):
        super(GradNormRepFunction, self).__init__()

    @staticmethod
    def forward(ctx, input):
        shape = input.shape
        ret = input.expand(GradNormRepFunction.n, *shape)
        return ret.clone()

    @staticmethod
    def backward(ctx, grad_output):
        global count
        num_grads = grad_output.shape[0]
        batch_size = grad_output.shape[1]
        grad_output = grad_output.float()
        if num_grads >= 2:

            GiW = [torch.sqrt(grad_output[i].reshape(-1).dot(grad_output[i].reshape(-1))) *
                   GradNormRepFunction.current_weights[i] for i in range(num_grads)]
            GW_bar = torch.mean(torch.stack(GiW))

            try:
                Li_ratio = [c / max(i, .0000001) for c, i in
                            zip(GradNormRepFunction.current_task_losses, GradNormRepFunction.inital_task_losses)]
                mean_ratio = statistics.mean(Li_ratio)
                ri = [lir / max(mean_ratio, .00000001) for lir in Li_ratio]
                target_grad = [float(GW_bar * (max(r_i, .00000001) ** 1.5)) for r_i in ri]

                target_weight = [float(target_grad[i] / float(GiW[i])) for i in range(num_grads)]
                total_weight = sum(target_weight)
                total_weight = max(.0000001, total_weight)
                target_weight = [i * num_grads / total_weight for i in target_weight]

                for i in range(len(GradNormRepFunction.current_weights)):
                    wi = GradNormRepFunction.current_weights[i]
                    GradNormRepFunction.current_weights[i] += (.0001 * wi if (wi < target_weight[i]) else -.0001 * wi)

                count += 1
                if count % 80 == 0:
                    with open("gradnorm_weights.txt", "a") as myfile:
                        myfile.write('target: ' + str(target_weight) + '\n')

                total_weight = sum(GradNormRepFunction.current_weights)
                total_weight = max(.0000001, total_weight)

                GradNormRepFunction.current_weights = [i * num_grads / total_weight for i in
                                                       GradNormRepFunction.current_weights]
            except:
                pass

            grad_outputs = [grad_output[i] * GradNormRepFunction.current_weights[i] * (1 / math.sqrt(num_grads)) for i
                            in range(num_grads)]
            output = grad_outputs[0]
            for i in range(1, num_grads):
                output += grad_outputs[i]
            return output.half()
        elif num_grads == 1:
            grad_input = grad_output.clone()
            out = grad_input.sum(dim=0)
        else:
            pass
        return out


gradnorm_rep_function = GradNormRepFunction.apply
