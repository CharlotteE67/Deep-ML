# -*- coding: utf-8 -*-            
# @Author : CharlottE
# @Time : 2024/11/7 14:32
"""
Implementing Basic Autograd Operations (medium)
Special thanks to Andrej Karpathy for making a video about this, if you haven't already check out his videos on YouTube https://youtu.be/VMj-3S1tku0?si=gjlnFP4o3JRN9dTg.
Write a Python class similar to the provided 'Value' class that implements the basic autograd operations: addition, multiplication,
and ReLU activation. The class should handle scalar values and should correctly compute gradients for these operations through automatic differentiation.
Example
Example:
        a = Value(2)
        b = Value(-3)
        c = Value(10)
        d = a + b * c
        e = d.relu()
        e.backward()
        print(a, b, c, d, e)
        Output: Value(data=2, grad=0) Value(data=-3, grad=10) Value(data=10, grad=-3) Value(data=-28, grad=1) Value(data=0, grad=1)
        Explanation: The output reflects the forward computation and gradients after backpropagation.
        The ReLU on 'd' zeros out its output and gradient due to the negative data value.
"""


class Value:
    def __init__(self, data, _children=(), _op=''):
        self.data = data
        self.grad = 0
        self._backward = lambda: None
        self._prev = set(_children)
        self._op = _op

    def __repr__(self):
        return f"Value(data={self.data}, grad={self.grad})"

    def __add__(self, other):
        # Implement addition here
        out = Value(self.data + other.data, {self, other}, "+")

        def _backward():
            self.grad += out.grad
            other.grad += out.grad

        out._backward = _backward
        return out

    def __mul__(self, other):
        # Implement multiplication here
        out = Value(self.data * other.data, {self, other}, "*")

        def _backward():
            self.grad += out.grad * other.data
            other.grad += out.grad * self.data

        out._backward = _backward
        return out

    def relu(self):
        # Implement ReLU here
        out = Value(max(0, self.data), {self}, "relu")

        def _backward():
            self.grad += (out.data > 0) * out.grad

        out._backward = _backward
        return out

    def backward(self):
        # Implement backward pass here
        topo = []
        visited = set()

        def build_topo(v):
            if v not in visited:
                visited.add(v)
                for child in v._prev:
                    build_topo(child)
                topo.append(v)

        build_topo(self)
        self.grad = 1
        for v in reversed(topo):
            v._backward()


a = Value(2)
b = Value(-3)
c = Value(10)
d = a + b * c
e = d.relu()
e.backward()
print(a, b, c, d, e)

"""
Test Case 1: Accepted
Input:
a = Value(2);b = Value(3);c = Value(10);d = a + b * c  ;e = Value(7) * Value(2);f = e + d;g = f.relu()  
g.backward()
print(a,b,c,d,e,f,g)
Output:
Value(data=2, grad=1) Value(data=3, grad=10) Value(data=10, grad=3) Value(data=32, grad=1) Value(data=14, grad=1) Value(data=46, grad=1) Value(data=46, grad=1)
Expected:
Value(data=2, grad=1) Value(data=3, grad=10) Value(data=10, grad=3) Value(data=32, grad=1) Value(data=14, grad=1) Value(data=46, grad=1) Value(data=46, grad=1)

class Value:
    def __init__(self, data, _children=(), _op=''):
        self.data = data
        self.grad = 0
        self._backward = lambda: None
        self._prev = set(_children)
        self._op = _op

    def __add__(self, other):
        other = other if isinstance(other, Value) else Value(other)
        out = Value(self.data + other.data, (self, other), '+')
        def _backward():
            self.grad += out.grad
            other.grad += out.grad
        out._backward = _backward
        return out

    def __mul__(self, other):
        other = other if isinstance(other, Value) else Value(other)
        out = Value(self.data * other.data, (self, other), '*')
        def _backward():
            self.grad += other.data * out.grad
            other.grad += self.data * out.grad
        out._backward = _backward
        return out

    def relu(self):
        out = Value(0 if self.data < 0 else self.data, (self,), 'ReLU')
        def _backward():
            self.grad += (out.data > 0) * out.grad
        out._backward = _backward
        return out

    def backward(self):
        topo = []
        visited = set()
        def build_topo(v):
            if v not in visited:
                visited.add(v)
                for child in v._prev:
                    build_topo(child)
                topo.append(v)
        build_topo(self)
        self.grad = 1
        for v in reversed(topo):
            v._backward()
    def __repr__(self):
        return f"Value(data={self.data}, grad={self.grad})"

"""