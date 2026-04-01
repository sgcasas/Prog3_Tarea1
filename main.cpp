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
    static Tensor zeros(const vector<size_t>& s) {

    }
  }
};

int main() {
  return 0;
}