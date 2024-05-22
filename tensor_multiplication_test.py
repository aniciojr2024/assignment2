from tensor_multiplication import (
    create_tensor_of_val,
    calculate_elementwise_product,
    calculate_matrix_product,
    calculate_matrix_prod_with_bias,
    calculate_activation,
    calculate_output
)
import torch
import pytest

def test_tensor_creation():
    res = create_tensor_of_val((2, 3), 3)
    print(f"Test Tensor Creation - Expected: tensor([[3., 3., 3.], [3., 3., 3.]]), Actual: {res}")
    assert res.shape == (2, 3), "shape is wrong"
    assert res.tolist() == [[3.0, 3.0, 3.0], [3.0, 3.0, 3.0]], "values are wrong"
    assert isinstance(res, torch.Tensor), "not a tensor"

def test_elementwise_product():
    A = torch.tensor([[1.0, 2.0, 3.0], [4.0, 5.0, 6.0]])
    B = torch.tensor([[0.5, 0.5, 0.5], [0.5, 0.5, 0.5]])
    res = calculate_elementwise_product(A, B)
    print(f"Test Elementwise Product - Expected: tensor([[0.5, 1.0, 1.5], [2.0, 2.5, 3.0]]), Actual: {res}")
    assert res.tolist() == [[0.5, 1.0, 1.5], [2.0, 2.5, 3.0]], "wrong output"
    assert isinstance(res, torch.Tensor), "not a tensor"

def test_elementwise_product_fails():
    A = torch.tensor([[1.0, 2.0, 3.0], [4.0, 5.0, 6.0]])
    B = torch.tensor([0.5, 0.5, 0.5, 1.0])
    with pytest.raises(RuntimeError):
        calculate_elementwise_product(A, B)

def test_product():
    X = torch.tensor([[1.0, 2.0, 3.0]])
    W = torch.tensor([[0.5, 0.5, 0.5]])
    res = calculate_matrix_product(X, W)
    print(f"Test Product - Expected: tensor([[3.0]]), Actual: {res}")
    assert res.tolist() == [[3.0]], "wrong output"
    assert isinstance(res, torch.Tensor), "not a tensor"

def test_product_fails():
    X = torch.tensor([[1.0, 2.0, 3.0, 4.0]])
    W = torch.tensor([[0.5, 0.5, 0.5], [1.0, 1.0, 1.0]])
    with pytest.raises(RuntimeError):
        calculate_matrix_product(X, W)

def test_neuron():
    """Test a single neuron"""
    X = torch.tensor([[1.0, 2.0, 3.0]])
    W = torch.tensor([[0.5, 0.5, 0.5]])
    b = torch.tensor(-1.5)
    res = calculate_matrix_prod_with_bias(X, W, b)
    print(f"Test Neuron - Expected: tensor([[1.5]]), Actual: {res}")
    assert res == torch.tensor(1.5), "wrong output"
    assert isinstance(res, torch.Tensor), "not a tensor"

    assert calculate_activation(torch.tensor(0.0)) == torch.tensor(0.0), "wrong activation"
    assert calculate_activation(torch.tensor(0.1)) == torch.tensor(1.0), "wrong activation"
    assert calculate_activation(torch.tensor(-0.1)) == torch.tensor(0.0), "wrong activation"
    assert isinstance(calculate_activation(torch.tensor(0.0)), torch.Tensor), "not a tensor"

    assert calculate_output(X, W, b) == torch.tensor(1.0), "wrong output"
    assert isinstance(calculate_output(X, W, b), torch.Tensor), "not a tensor"

def test_a_batch_of_inputs():
    """Test a batch of inputs"""
    X = torch.tensor([[1.0, 2.0, 3.0], [4.0, 5.0, 6.0]])
    W = torch.tensor([[0.5, 0.5, 0.5]])
    b = torch.tensor(-1.5)
    res = calculate_matrix_prod_with_bias(X, W, b)
    print(f"Test Batch of Inputs - Expected: tensor([[1.5], [6.0]]), Actual: {res}")
    assert res.tolist() == [[1.5], [6.0]], "wrong output"

def test_two_neurons():
    """Test two neurons"""
    X = torch.tensor([[1.0, 2.0, 3.0]])
    W = torch.tensor([[0.5, 0.5, 0.5], [1.0, 1.0, 1.0]])
    b = torch.tensor([-1.5, -0.5])
    res = calculate_matrix_prod_with_bias(X, W, b)
    print(f"Test Two Neurons - Expected: tensor([[1.5, 5.5]]), Actual: {res}")
    assert res.tolist() == [[1.5, 5.5]], "wrong output"
    assert isinstance(res, torch.Tensor), "not a tensor"
    assert calculate_output(X, W, b).tolist() == [[1.0, 1.0]], "wrong output"
    assert isinstance(calculate_output(X, W, b), torch.Tensor), "not a tensor"

def test_two_neurons_with_a_batch_of_inputs():
    """Test two neurons with a batch of inputs"""
    X = torch.tensor([[1.0, 2.0, 3.0], [4.0, 5.0, 6.0]])
    W = torch.tensor([[0.5, 0.5, 0.5], [1.0, 1.0, 1.0]])
    b = torch.tensor([-1.5, -0.5])
    res = calculate_matrix_prod_with_bias(X, W, b)
    print(f"Test Two Neurons with a Batch of Inputs - Expected: tensor([[1.5, 5.5], [6.0, 14.5]]), Actual: {res}")
    assert res.tolist() == [[1.5, 5.5], [6.0, 14.5]], "wrong output"
    assert calculate_output(X, W, b).tolist() == [[1.0, 1.0], [1.0, 1.0]], "wrong output"

if __name__ == "__main__":
    test_tensor_creation()
    test_elementwise_product()
    test_elementwise_product_fails()
    test_product()
    test_product_fails()
    test_neuron()
    test_a_batch_of_inputs()
    test_two_neurons()
    test_two_neurons_with_a_batch_of_inputs()
    print("All tests passed!")
