#include <iostream>
#include <vector>
#include <algorithm>
#include <cmath>
using namespace std;

class TensorTransform;

class Tensor {
  vector<size_t> shape;
  double* values;
public:
  //constructor
  Tensor(const vector<size_t>& s, const vector<double>& v) {
    shape = s;
    size_t tam = shape.size();
    int tot = 1;
    if (shape.empty() || tam > 3 ) throw std::invalid_argument("El tensor debe tener máximo 3 dimensiones");
    for (int i = 0; i < tam; i++) { tot *= shape[i]; }
    if (tot != v.size()) throw std::invalid_argument("La cantidad de valores no coincide con el tamaño del tensor");
    values = new double[tot];
    for (int i = 0; i < tot; i++) { values[i] = v[i]; }
  }
  //constructor copia
  Tensor(const Tensor& other) {
    shape = other.shape;
    int tot = 1;
    for (int i = 0; i < shape.size(); i++) { tot *= shape[i]; }
    values = new double[tot];
    for (int i = 0; i < tot; i++) { values[i] = other.values[i]; }
  }
  //constructor de movimiento
  Tensor(Tensor&& other) noexcept {
    shape = other.shape;
    values = other.values;
    other.values = nullptr;
    other.shape.clear();
  }
  //asignador de copia
  Tensor& operator=(const Tensor& other) {
    if (this == &other) return *this;
    delete[] values;
    shape = other.shape;
    int tot = 1;
    for (int i = 0; i < shape.size(); i++) { tot *= shape[i]; }
    values = new double[tot];
    for (int i = 0; i < tot; i++) { values[i] = other.values[i]; }
    return *this;
  }
  //asignacion de movimiento
  Tensor& operator=(Tensor&& other) noexcept {
    delete [] values;
    shape = other.shape;
    values = other.values;
    other.values = nullptr;
    other.shape.clear();
    return *this;
  }
  //destructor
  ~Tensor() {
    delete [] values;
  }
  double* getValues() const { return values; }
  vector<size_t> getShape() const { return shape; }
  //declaracion apply
  Tensor apply(const TensorTransform& t) const;
  static Tensor zeros(const vector<size_t>& s) {
    if (s.empty() || s.size() > 3 ) throw std::invalid_argument("El tensor debe tener máximo 3 dimensiones");
    int tot = 1;
    for (int i = 0; i < s.size(); i++) { tot *= s[i]; }
    return Tensor(s, vector<double>(tot, 0));
  }
  static Tensor ones(const vector<size_t>& s) {
    if (s.empty() || s.size() > 3 ) throw std::invalid_argument("El tensor debe tener máximo 3 dimensiones");
    int tot = 1;
    for (int i = 0; i < s.size(); i++) { tot *= s[i]; }
    return Tensor(s, vector<double>(tot, 1));
  }
  static Tensor random(const vector<size_t>& s, double min, double max) {
    if (s.empty() || s.size() > 3 ) throw std::invalid_argument("El tensor debe tener máximo 3 dimensiones");
    int tot = 1;
    for (int i = 0; i < s.size(); i++) { tot *= s[i]; }
    vector<double> v(tot);
    for (int i = 0; i < tot; i++) { v[i] = min + (rand() % 1000) / 1000.0 * (max - min); }
    return Tensor(s, v);
  }
  static Tensor arange(int start, int end) {
    int tam = end - start;
    vector<double> v(tam);
    for (int i = start, j = 0; i < end; i++, j++) { v[j] = i; }
    return Tensor({v.size()},v);
  }
};

class TensorTransform {
public:
  virtual Tensor apply(const Tensor& t) const = 0;
  virtual ~TensorTransform() = default;
};

class ReLu : public TensorTransform {
public:
  Tensor apply(const Tensor& t) const override {
    int tot = 1;
    vector<size_t> shape = t.getShape();
    for (int i = 0; i < shape.size(); i++) { tot *= shape[i]; }
    vector<double> val(tot);
    for (int i = 0; i < tot; i++) { val[i] = max(0.0, t.getValues()[i]); }
    return Tensor(shape, val);
  }
};

class Sigmoid : public TensorTransform {
public:
  Tensor apply(const Tensor& t) const override {
    int tot = 1;
    vector<size_t> shape = t.getShape();
    for (int i = 0; i < shape.size(); i++) { tot *= shape[i]; }
    vector<double> val(tot);
    for (int i = 0; i < tot; i++) { val[i] = 1.0 / (1.0 + exp(-t.getValues()[i])); }
    return Tensor(shape, val);
  }
};

//implementacion apply
Tensor Tensor::apply(const TensorTransform &t) const {
  return t.apply(*this);
}

//implementacion de operadores
Tensor Tensor::operator+(const Tensor& other) const {
  if (shape != other.shape) throw std::invalid_argument("Incompatibles");
  int tot = 1;
  for (size_t i = 0; i < shape.size(); i++) tot *= shape[i];
  vector<double> result(tot);
  for (int i = 0; i < tot; i++) result[i] = values[i] + other.values[i];
  return Tensor(shape, result);
}

Tensor Tensor::operator-(const Tensor& other) const {
  if (shape != other.shape) throw std::invalid_argument("Incompatibles");
  int tot = 1;
  for (size_t i = 0; i < shape.size(); i++) tot *= shape[i];
  vector<double> result(tot);
  for (int i = 0; i < tot; i++) result[i] = values[i] - other.values[i];
  return Tensor(shape, result);
}

Tensor Tensor::operator*(const Tensor& other) const {
  if (shape != other.shape) throw std::invalid_argument("Incompatibles");
  int tot = 1;
  for (size_t i = 0; i < shape.size(); i++) tot *= shape[i];
  vector<double> result(tot);
  for (int i = 0; i < tot; i++) result[i] = values[i] * other.values[i];
  return Tensor(shape, result);
}

Tensor Tensor::operator*(double scalar) const {
  int tot = 1;
  for (size_t i = 0; i < shape.size(); i++) tot *= shape[i];
  vector<double> result(tot);
  for (int i = 0; i < tot; i++) result[i] = values[i] * scalar;
  return Tensor(shape, result);
}

//modificacion de dimensiones
Tensor Tensor::view(const vector<size_t>& newshape) const {
  if (newshape.empty() || newshape.size() > 3) throw std::invalid_argument("El tensor debe tener máximo 3 dimensiones");
  int tot = 1;
  for (auto s : shape) tot *= s;
  int newtot = 1;
  for (auto s : newshape) newtot *= s;
  if (tot != newtot) throw std::invalid_argument("La cantidad de valores no coincide con el tamaño del tensor");
  Tensor t = *this;
  t.shape = newshape;
  return t;
}
Tensor Tensor::unsqueeze(size_t dim) const {
  if (shape.size() >= 3) throw std::invalid_argument("El tensor debe tener máximo 3 dimensiones");
  if (dim > shape.size()) throw std::invalid_argument("Dimensión inválida");
  vector<size_t> newshape;
  for (size_t i = 0; i < shape.size() + 1; i++) {
    if (i == dim) newshape.push_back(1);
    else newshape.push_back(shape[i < dim ? i : i - 1]);
  }
  int tot = 1;
  for (auto s : shape) tot *= s;
  vector<double> v(tot);
  for (int i = 0; i < tot; i++) v[i] = values[i];
  return Tensor(newshape, v);
}

//Concatenacion

Tensor Tensor::concat(const vector<Tensor>& tensors, size_t dim) {
  if (tensors.empty()) throw std::invalid_argument("Lista vacía");
  vector<size_t> shapebase = tensors[0].shape;
  size_t dims = shapebase.size();
  if (dims == 0 || dims > 3) throw std::invalid_argument("Dimensiones inválidas");
  if (dim >= dims) throw std::invalid_argument("Dimensión inválida");
  for (const auto& t : tensors) {
    if (t.shape.size() != dims) throw std::invalid_argument("Dimensiones incompatibles");
    for (size_t i = 0; i < dims; i++) {
      if (i != dim && t.shape[i] != shapebase[i]) throw std::invalid_argument("Incompatibles");
    }
  }
  vector<size_t> newshape = shapebase;
  newshape[dim] = 0;
  for (const auto& t : tensors) {
    newshape[dim] += t.shape[dim];
  }
  int total = 1;
  for (auto s : newshape) total *= s;
  vector<double> result(total);
  int aux = 0;
  for (const auto& t : tensors) {
    int t_total = 1;
    for (auto s : t.shape) t_total *= s;
    for (int i = 0; i < t_total; i++) result[aux + i] = t.values[i];
    aux += t_total;
  }
  return Tensor(newshape, result);
}

//Funciones amigas

Tensor dot(const Tensor& a, const Tensor& b) {
  vector<size_t> shapeA = a.getShape();
  vector<size_t> shapeB = b.getShape();
  if (shapeA[0] != shapeB[0]) throw std::invalid_argument("Incompatibles");
  int n = shapeA[0];
  double sum = 0.0;
  double* valA = a.getValues();
  double* valB = b.getValues();
  for (int i = 0; i < n; i++) sum += valA[i] * valB[i];
  return Tensor({1}, {sum});
}
Tensor matmul(const Tensor& A, const Tensor& B) {
  vector<size_t> shapeA = A.getShape();
  vector<size_t> shapeB = B.getShape();
  if (shapeA.size() != 2 || shapeB.size() != 2) throw std::invalid_argument("Solo matrices 2D");
  int f1 = shapeA[0];
  int c1 = shapeA[1];
  int f2 = shapeB[0];
  int c2 = shapeB[1];
  if (c1 != f2) throw std::invalid_argument("Dimensiones incompatibles");
  vector<double> result(f1 * c2, 0.0);
  double* a = A.getValues();
  double* b = B.getValues();
  for (int i = 0; i < f1; i++) {
    for (int j = 0; j < c2; j++) {
      double sum = 0.0;
      for (int k = 0; k < c1; k++) {
        sum += a[i * c1 + k] * b[k * c2 + j];
      }
      result[i * c2 + j] = sum;
    }
  }
  return Tensor({(size_t)f1, (size_t)c2}, result);
}

int main() {
  Tensor T = Tensor::random({1000, 20, 20}, 0.0, 1.0);
  Tensor T_trans = T.view({1000, 400});
  Tensor W1 = Tensor::random({400, 100}, -1.0, 1.0);
  Tensor Mult = matmul(T_trans, W1);
  Tensor b1 = Tensor::random({1, 100}, -1.0, 1.0);
  vector<double> temp1(1000 * 100);
  double* Mult1 = Mult.getValues();
  double* bias = b1.getValues();
  for (int i = 0; i < 1000; i++) {
    for (int j = 0; j < 100; j++) {
      temp1[i * 100 + j] = Mult1[i * 100 + j] + bias[j];
    }
  }
  Tensor A1({1000, 100}, temp1);
  ReLu relu;
  Tensor H1 = A1.apply(relu);
  Tensor W2 = Tensor::random({100, 10}, -1.0, 1.0);
  Tensor b2 = Tensor::random({1, 10}, -1.0, 1.0);
  Tensor Z2 = matmul(H1, W2);
  vector<double> temp2(1000 * 10);
  double* z2 = Z2.getValues();
  double* bias2 = b2.getValues();
  for (int i = 0; i < 1000; i++) {
    for (int j = 0; j < 10; j++) {
      temp2[i * 10 + j] = z2[i * 10 + j] + bias2[j];
    }
  }
  Tensor A2({1000, 10}, temp2);
  Sigmoid sigmoid;
  Tensor output = A2.apply(sigmoid);
  return 0;
}
