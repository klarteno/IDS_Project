import numpy as np
import torch
import matplotlib.pyplot as plt



def plot_scatter_values(data:list, label_y = 'data'):
    plt.plot(data, color='blue', marker='o',mfc='black', linestyle='dashed', ) #plot the data
    plt.xticks(range(0,len(data), 1)) #set the tick frequency on x-axis

    plt.ylabel(label_y) #set the label for y axis
    plt.xlabel('epochs') #set the label for x-axis
    plt.title("Plotting "+label_y) #set the title of the graph
    plt.show() #display the graph
    

# plot float values of the form : 0.0004 
def plot_float_values(values, label_y = 'Scheduler learning history'):
    import plotly.graph_objects as go
    import plotly.io as pio
    pio.renderers.default = 'notebook+jupyterlab'
    fig = go.Figure()
    fig.add_trace(go.Scatter(x=list(range(0, len(values))), y=values, mode='lines+markers', name='lines+markers'))
    fig.update_layout(title='Plot of '+label_y, xaxis_title='x', yaxis_title=label_y)
    fig.show()
    
    

def plot_trainning_eval(
    minibatch_losses,
    num_epochs,
    averaging_iterations=100,
    type_plot="Loss",
    custom_label="",
):

    iter_per_epoch = len(minibatch_losses) // num_epochs

    plt.figure()
    ax1 = plt.subplot(1, 1, 1)
    ax1.plot(
        range(len(minibatch_losses)),
        (minibatch_losses),
        label=f"Minibatch {type_plot} {custom_label}",
    )
    ax1.set_xlabel("Iterations")
    ax1.set_ylabel(f" {type_plot}")

    if len(minibatch_losses) < 1000:
        num_losses = len(minibatch_losses) // 2
    else:
        num_losses = 1000

    ax1.set_ylim([0, np.max(minibatch_losses[num_losses:]) * 1.5])

    ax1.plot(
        np.convolve(
            minibatch_losses,
            np.ones(
                averaging_iterations,
            )
            / averaging_iterations,
            mode="valid",
        ),
        label=f"Running Average{custom_label}",
    )
    ax1.legend()

    ###################
    # Set scond x-axis
    ax2 = ax1.twiny()
    newlabel = list(range(num_epochs + 1))

    newpos = [e * iter_per_epoch for e in newlabel]

    ax2.set_xticks(newpos[::10])
    ax2.set_xticklabels(newlabel[::10])

    ax2.xaxis.set_ticks_position("bottom")
    ax2.xaxis.set_label_position("bottom")
    ax2.spines["bottom"].set_position(("outward", 45))
    ax2.set_xlabel("Epochs")
    ax2.set_xlim(ax1.get_xlim())
    ###################

    plt.tight_layout()


def plot_accuracy(train_acc, valid_acc):

    num_epochs = len(train_acc)

    plt.plot(np.arange(1, num_epochs + 1), train_acc, label="Training")
    plt.plot(np.arange(1, num_epochs + 1), valid_acc, label="Validation")

    plt.xlabel("Epoch")
    plt.ylabel("Accuracy")
    plt.legend()

    plt.tight_layout()


def plot_multiple_trainning_evals(
    evals_list, num_epochs, averaging_iterations=100, custom_labels_list=None
):

    for i, _ in enumerate(evals_list):
        if not len(evals_list[i]) == len(evals_list[0]):
            raise ValueError(
                "All loss tensors need to have the same number of elements."
            )

    if custom_labels_list is None:
        custom_labels_list = [str(i) for i, _ in enumerate(custom_labels_list)]

    iter_per_epoch = len(evals_list[0]) // num_epochs

    plt.figure()
    ax1 = plt.subplot(1, 1, 1)

    for i, minibatch_loss_tensor in enumerate(evals_list):
        ax1.plot(
            range(len(minibatch_loss_tensor)),
            (minibatch_loss_tensor),
            label=f"Minibatch Loss{custom_labels_list[i]}",
        )
        ax1.set_xlabel("Iterations")
        ax1.set_ylabel("Loss")

        ax1.plot(
            np.convolve(
                minibatch_loss_tensor,
                np.ones(
                    averaging_iterations,
                )
                / averaging_iterations,
                mode="valid",
            ),
            color="black",
        )

    if len(evals_list[0]) < 1000:
        num_losses = len(evals_list[0]) // 2
    else:
        num_losses = 1000
    maxes = [np.max(evals_list[i][num_losses:]) for i, _ in enumerate(evals_list)]
    ax1.set_ylim([0, np.max(maxes) * 1.5])
    ax1.legend()

    ###################
    # Set scond x-axis
    ax2 = ax1.twiny()
    newlabel = list(range(num_epochs + 1))

    newpos = [e * iter_per_epoch for e in newlabel]

    ax2.set_xticks(newpos[::10])
    ax2.set_xticklabels(newlabel[::10])

    ax2.xaxis.set_ticks_position("bottom")
    ax2.xaxis.set_label_position("bottom")
    ax2.spines["bottom"].set_position(("outward", 45))
    ax2.set_xlabel("Epochs")
    ax2.set_xlim(ax1.get_xlim())
    ###################

    plt.tight_layout()


def transferTensorsToDevice(tensors, device="cpu"):
    return [tensor.to(device) for tensor in tensors]


def _plotEvaluationResults(
    accuracies_scores,
    loss_scores,
    f1_scores,
    auroc_scores,
    label="MLP_ver1",
):
    plot_multiple_trainning_evals(
        evals_list=(accuracies_scores, loss_scores),
        num_epochs=len(accuracies_scores),
        custom_labels_list=(" --" + label + " Accuracies vs Losses", " --" + label + " Losses"),
    )

    plot_trainning_eval(
        accuracies_scores,
        num_epochs=len(accuracies_scores),
        type_plot="Accuracies",
        custom_label=" " + label,
    )
    plot_trainning_eval(
        loss_scores, num_epochs=len(loss_scores), type_plot="Losses", custom_label=" " + label
    )

    plot_trainning_eval(
        f1_scores,
        num_epochs=len(f1_scores),
        type_plot=" F1_Scores",
        custom_label=" " + label,
    )
    plot_trainning_eval(
        auroc_scores,
        num_epochs=len(auroc_scores),
        type_plot=" Auroc_Scores",
        custom_label=" " + label,
    )



def _plotEvaluationResults2(
    accuracies_scores,
    loss_scores,
    f1_scores,
    auroc_scores
):
    plt.subplot(211)             # the first subplot in the first figure
    plot_scatter_values(accuracies_scores, label_y = 'accuracies')
    plt.subplot(212) 
    plot_scatter_values(loss_scores, label_y = 'losses')
    plt.subplot(211) 
    plot_scatter_values(f1_scores, label_y = 'F1-scores')
    plt.subplot(212) 
    plot_scatter_values(auroc_scores, label_y = 'Auroc_scores')
    
    
def plotEvaluationResults(
    accuracies_scores,
    loss_scores,
    f1_scores,
    auroc_scores,
    label="MLP_ver1",
):
    if type(accuracies_scores[0]) is torch.Tensor:
        accuracies_scores = transferTensorsToDevice(accuracies_scores, device="cpu")
        
    elif type(loss_scores[0]) is torch.Tensor:  
        loss_scores = transferTensorsToDevice(loss_scores, device="cpu")
    
    elif type(f1_scores[0]) is torch.Tensor:  
        f1_scores = transferTensorsToDevice(f1_scores, device="cpu")
    
    elif type(auroc_scores[0]) is torch.Tensor:  
        auroc_scores = transferTensorsToDevice(auroc_scores, device="cpu")

    _plotEvaluationResults2(
        accuracies_scores, loss_scores, f1_scores, auroc_scores
    )
