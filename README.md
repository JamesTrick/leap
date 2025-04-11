## LEAP

Leap is a fun exploratory project from James Malcolm. Leap introduces one main contribution to the ML community. 
Notably 'leaping ahead', predicting the end token of a sequence, then in-filling the text in-between tokens.

## Motivation

The motivation for this is to try to gain two things:

* More coherent, less waffling text. By having the context, and the predicted end token, do we get less-error prone text?
* More natural, flowing text. With varied sentence structure. This could be done by the actual distance prediction.
Which in essence, plans how long a sentence should be and fills the sentence accordingly.

The leaping approach differs slightly to traditional autoregressive models which predict token-by-token and may encourage 
the model to 'plan-ahead'.

Despite this, I doubt it's a practical approach for two reasons:

* It effectively adds computational overload. Instead of making a single prediction of the text token, you're making two. 
The distant token, then the tokens in between in an autoregressive manner.
* Whilst we may get _less_ error-prone text. I expect when we get errors/hallucinations, that these will be exacerbated by the 
distant token acting as an anchor point

Nonetheless, it's an interesting approach compared to autoregressive models and no harm in trying.

## Implementation

The implementation of these models, dataloaders and training is done in [PyTorch](https://pytorch.org), with 
[Ray](https://docs.ray.io/) to allow for easier distributed training.

## Models

### Baseline LSTM

This baseline model is a straight-forward model primarily designed to test the dataloaders and training loop. It's
comprised of a single embedding layer, single LSTM layer and two fully-connected layers.

It does perform 'sequential' prediction, by predicting the distant target token first, and uses the logit output of 
that to predict the gap tokens, as shown below:

```python
class LeapBaselineModel(nn.Module):
    ...
    
    def forward(self, inputs):
        ...
        target_output = self.fc_target(final_hidden_state.squeeze(1))
        target_state_expanded = final_hidden_state.expand(-1, lstm_out.size(1), -1)
        gap_input_features = torch.cat([lstm_out, target_state_expanded], dim=-1)
        gap_output = self.fc_gap(gap_input_features)
        ...
```

### LEAP using Transformers

This is more of a 'modern' solution, and it implements the same model using a fairly straight-forward. Transformer 
structure.

It's comprised of both token and positional embeddings. Context is encoded through a series of TransformerEncoder layers 
and finally passed through a vanilla Attention layer, before being used to predict the target / distant token.

Although in the generate function, I've added temperature and topk parameters. These are not used in the forward/model 
training.

## Training

A training function is located in `src/train.py`, it uses [Ray Train](https://docs.ray.io/en/latest/train/train.html) for 
distributed training. The current implementation is tested to handle both CPU and GPU training with no additional changes.


## Evaluation

Evaluation can be done in two parts. During training, losses are tracked using a crossentropy loss. This makes sense as
we are trying to 'classify' the token against a ground truth.

More extensive evaluation can be done post-training such a tests against common benchmarks through something like 
[EleutherAI's evaluation harness](https://github.com/EleutherAI/lm-evaluation-harness/tree/main), or perhaps more 
task-specific evaluations.

## Improvements

To date, the model doesn't predict the sentence length. Intuitively, I think adding sentence length prediction would 
lead to more varied and natural sentence structure.

An avenue to explore would be to a BERT style LM head to predict the gaps. BERT specialises in filling masked tokens. 
Given we have context on the left, and a predicted token on the right, the task could be reframed as masked language 
prediction. This may, lead to increased performance.