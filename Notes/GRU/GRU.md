# GRU 

## Gated Recurrent Unit (GRU) by Cho et al. (2014)

### Introduction

The GRU is a simplified version of the LSTM. It has fewer parameters than LSTM, as it lacks an output gate. GRU’s performance on certain tasks of polyphonic music modeling and speech signal modeling was found to be similar to that of LSTM.

### Architecture

GRU consists of two gates: reset gate and update gate. 

#### Reset gate

The reset gate decides how to combine the new input with the previous memory. If we set the reset to all 0’s, it ignores the previous memory completely. If we set it to all 1’s, it ignores the new input completely.

$$ r_t = \sigma(W_{rx} x_t + W_{rh} h_{t-1}) $$

#### Update gate

The update gate decides how much of the previous state to keep around. If we set the update to all 0’s, it ignores the previous state completely. If we set it to all 1’s, it ignores the new input completely.

$$ z_t = \sigma(W_{zx} x_t + W_{zh} h_{t-1}) $$

#### New memory content

The new memory content is a combination of the previous memory and the memory we’re going to add. First, we multiply the previous state by the reset gate. This decides how much of the previous state we’re going to keep around. Then we multiply the new input by 1 - update gate. This decides how much of the new state is going to be added to the memory. Finally, we add the two together to get the new memory.

$$ \tilde{h}_t = \tanh(W_{hx} x_t + W_{hh} (r_t \odot h_{t-1})) $$
$$ h_t = (1 - z_t) \odot h_{t-1} + z_t \odot \tilde{h}_t $$

#### Summary

- If we set the reset to all 0’s and update gate to all 1’s we arrive at LSTM. 

$$ r_t = 0  \text{ and }  z_t = 1 $$

$$ h_t = h_{t-1} + \tilde{h}_t $$
$$ h_t = h_{t-1} + \tanh(W_{hx} x_t) $$

- If we set the reset to all 1’s and update gate to all 0’s we arrive at a simple RNN.

$$ r_t = 1  \text{ and }  z_t = 0 $$

$$ h_t = \tanh(W_{hx} x_t + W_{hh} h_{t-1}) $$


## Advantage of these gated units (GRU and LSTM)

- First, it is easy for each unit to remember the existence of a specific feature in the input stream for a long series of steps. Any important feature, decided by either the forget gate of the LSTM unit or the update gate of the GRU, will not be overwritten but be maintained as it is.

- Second, and perhaps more importantly, this addition effectively creates shortcut paths that bypass multiple temporal steps. These shortcuts allow the error to be back-propagated easily without too quickly vanishing (if the gating unit is nearly saturated at 1) as a result of passing through multiple, bounded nonlinearities, thus reducing the difficulty due to vanishing gradients 

## Difference between GRU and LSTM

- GRU has two gates (reset and update), while LSTM has three gates (input, output and forget).

- One feature of the LSTM unit that is missing from the GRU is the controlled exposure of the memory content. In the LSTM unit, the amount of the memory content that is seen, or used by other units in the network is controlled by the output gate. On the other hand the GRU exposes its full content without any control.

$$ \text{LSTM: } h_t = o_t \odot \tanh(c_t) $$
$$ \text{GRU: } h_t = (1 - z_t) \odot h_{t-1} + z_t \odot \tilde{h}_t $$

- Another difference is in the location of the input gate, or the corresponding reset gate. The LSTM unit computes the new memory content without any separate control of the amount of information flowing from the previous time step. Rather, the LSTM unit controls the amount of the new memory content being added to the memory cell independently from the forget gate. On the other hand, the GRU controls the information flow from the previous activation when computing the new, candidate activation, but does not independently control the amount of the candidate activation being added (the control is tied via the update gate).

$$ \text{LSTM: } \tilde{c}_t = \tanh(W_{cx} x_t + W_{ch} h_{t-1}) $$
$$ \text{GRU: } \tilde{h}_t = \tanh(W_{hx} x_t + W_{hh} (r_t \odot h_{t-1})) $$

