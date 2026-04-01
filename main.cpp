#include <iostream>
#include <vector>
using namespace std;

class Tensor {
  vector<size_t> shape;
  vector<double> values;
public:
  Tensor(const vector<size_t>& s, const vector<double>& v) {
    shape = s;
    values = v;
    size_t tam = shape.size();
    int tot = 1;
    if (shape.empty() || tam > 3 ) throw std::invalid_argument("El tensor debe tener máximo 3 dimensiones");
    for (int i = 0; i < tam; i++) { tot *= shape[i]; }
    if (tot != values.size()) throw std::invalid_argument("La cantidad de valores no coincide con el tamaño del tensor");
  }
  void set_values(vector<double>& v) { values = v; }
  void set_shape(vector<size_t>& s) { shape = s; }
  static Tensor zeros(const vector<size_t>& s);
};

Tensor Tensor::zeros(const vector<size_t>& s) {
  int tot = 1;
  for (int i = 0; i < s.size(); i++) { tot *= s[i]; }
  return Tensor(s, vector<double>(tot, 0));
}

int main() {
  return 0;
}