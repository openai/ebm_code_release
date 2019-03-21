import tensorflow as tf
import numpy as np

from tensorflow.python.platform import flags
flags.DEFINE_bool('proposal_debug', False, 'Print hmc acceptance raes')

FLAGS = flags.FLAGS

def kinetic_energy(velocity):
    """Kinetic energy of the current velocity (assuming a standard Gaussian)
        (x dot x) / 2

    Parameters
    ----------
    velocity : tf.Variable
        Vector of current velocity

    Returns
    -------
    kinetic_energy : float
    """
    return 0.5 * tf.square(velocity)

def hamiltonian(position, velocity, energy_function):
    """Computes the Hamiltonian of the current position, velocity pair

    H = U(x) + K(v)

    U is the potential energy and is = -log_posterior(x)

    Parameters
    ----------
    position : tf.Variable
        Position or state vector x (sample from the target distribution)
    velocity : tf.Variable
        Auxiliary velocity variable
    energy_function
        Function from state to position to 'energy'
         = -log_posterior

    Returns
    -------
    hamitonian : float
    """
    batch_size = tf.shape(velocity)[0]
    kinetic_energy_flat = tf.reshape(kinetic_energy(velocity), (batch_size, -1))
    return tf.squeeze(energy_function(position)) + tf.reduce_sum(kinetic_energy_flat, axis=[1])

def leapfrog_step(x0,
                  v0,
                  neg_log_posterior,
                  step_size,
                  num_steps):

    # Start by updating the velocity a half-step
    v = v0 - 0.5 * step_size * tf.gradients(neg_log_posterior(x0), x0)[0]

    # Initalize x to be the first step
    x = x0 + step_size * v

    for i in range(num_steps):
        # Compute gradient of the log-posterior with respect to x
        gradient = tf.gradients(neg_log_posterior(x), x)[0]

        # Update velocity
        v = v - step_size * gradient

        # x_clip = tf.clip_by_value(x, 0.0, 1.0)
        # x = x_clip
        # v_mask = 1 - 2 * tf.abs(tf.sign(x - x_clip))
        # v = v * v_mask

        # Update x
        x = x + step_size * v

        # x = tf.clip_by_value(x, -0.01, 1.01)

    # x = tf.Print(x, [tf.reduce_min(x), tf.reduce_max(x), tf.reduce_mean(x)])

    # Do a final update of the velocity for a half step
    v = v - 0.5 * step_size * tf.gradients(neg_log_posterior(x), x)[0]

    # return new proposal state
    return x, v

def hmc(initial_x,
        step_size,
        num_steps,
        neg_log_posterior):
    """Summary

    Parameters
    ----------
    initial_x : tf.Variable
        Initial sample x ~ p
    step_size : float
        Step-size in Hamiltonian simulation
    num_steps : int
        Number of steps to take in Hamiltonian simulation
    neg_log_posterior : str
        Negative log posterior (unnormalized) for the target distribution

    Returns
    -------
    sample :
        Sample ~ target distribution
    """

    v0 = tf.random_normal(tf.shape(initial_x))
    x, v = leapfrog_step(initial_x,
                      v0,
                      step_size=step_size,
                      num_steps=num_steps,
                      neg_log_posterior=neg_log_posterior)

    orig = hamiltonian(initial_x, v0, neg_log_posterior)
    current = hamiltonian(x, v, neg_log_posterior)

    prob_accept = tf.exp(orig - current)

    if FLAGS.proposal_debug:
        prob_accept = tf.Print(prob_accept, [tf.reduce_mean(tf.clip_by_value(prob_accept, 0, 1))])

    uniform = tf.random_uniform(tf.shape(prob_accept))
    keep_mask = (prob_accept > uniform)
    # print(keep_mask.get_shape())

    x_new = tf.where(keep_mask, x, initial_x)
    return x_new
