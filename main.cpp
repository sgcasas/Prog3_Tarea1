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

int main() {
  return 0;
}