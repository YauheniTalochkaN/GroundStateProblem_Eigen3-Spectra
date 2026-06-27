#include <iostream>
#include <fstream>
#include <vector>
#include <unordered_map>
#include <utility>
#include <algorithm>
#include <numeric>
#include <complex>
#include <omp.h>

#include <Eigen/Core>
#include <Eigen/SparseCore>
#include <Spectra/HermEigsSolver.h>

using complexType = std::complex<double>;

struct Edge 
{
    Edge(int64_t u_, int64_t v_, double Jx_, double Jy_, double Jz_) : 
         u(u_), v(v_), Jx(Jx_), Jy(Jy_), Jz(Jz_) 
    {

    };

    ~Edge() = default;
    
    int64_t u, v;
    double Jx, Jy, Jz;
};

inline int64_t get_bit(int64_t state, int64_t i) 
{ 
    return (state >> i) & 1; 
}

inline int64_t get_flipped_state(int64_t state, int64_t i)
{
    return state ^ (1 << i);
}

inline int64_t get_flipped_state(int64_t state, int64_t i, int64_t j)
{
    return state ^ (1 << i) ^ (1 << j);
}

void build_sparse_hamiltonian(int64_t N, int64_t dim,   
                              const std::vector<Edge>& edges,
                              Eigen::SparseMatrix<complexType, Eigen::RowMajor, int64_t>& H1z,
                              Eigen::SparseMatrix<complexType, Eigen::RowMajor, int64_t>& H) 
{       
    for (int64_t state = 0; state < dim; ++state) 
    {
        for (int64_t i = 0; i < N; ++i) 
        {
            int64_t bit_i = get_bit(state, i);

            complexType sign_i = (bit_i == 0) ? complexType(1.0, 0.0) : complexType(-1.0, 0.0);

            H1z.coeffRef(state, state) += complexType(0.5, 0.0) * sign_i;
        }
        
        for (const auto& edge : edges) 
        {
            int64_t i = edge.u;
            int64_t j = edge.v;

            double Jx = edge.Jx / 4.0;
            double Jy = edge.Jy / 4.0;
            double Jz = edge.Jz / 4.0;
            
            int64_t bit_i = get_bit(state, i);
            int64_t bit_j = get_bit(state, j);

            int64_t flipped_state = get_flipped_state(state, i, j);

            complexType sign_i = (bit_i == 0) ? complexType(1.0, 0.0) : complexType(-1.0, 0.0);
            complexType sign_j = (bit_j == 0) ? complexType(1.0, 0.0) : complexType(-1.0, 0.0);
            
            H.coeffRef(flipped_state, state) += complexType(Jx, 0.0) + complexType(-Jy, 0.0) * sign_i * sign_j;
            
            H.coeffRef(state, state) += complexType(Jz, 0.0) * sign_i * sign_j;
        }
    }
}

template <typename Scalar_, int Flags = Eigen::RowMajor, typename StorageIndex = int64_t>
class CustomSparseHermMatProd
{
public:
    using Scalar = Scalar_;

private:
    using Index = Eigen::Index;
    using Vector = Eigen::Matrix<Scalar, Eigen::Dynamic, 1>;
    using MapConstVec = Eigen::Map<const Vector>;
    using MapVec = Eigen::Map<Vector>;
    using Matrix = Eigen::Matrix<Scalar, Eigen::Dynamic, Eigen::Dynamic>;
    using SparseMatrix = Eigen::SparseMatrix<Scalar, Flags, StorageIndex>;
    using ConstGenericSparseMatrix = const Eigen::Ref<const SparseMatrix>;

    ConstGenericSparseMatrix m_mat;

public:
    template <typename Derived>
    CustomSparseHermMatProd(const Eigen::SparseMatrixBase<Derived>& mat) :
        m_mat(mat)
    {
        static_assert(
            static_cast<int>(Derived::PlainObject::IsRowMajor) == static_cast<int>(SparseMatrix::IsRowMajor),
            "CustomSparseHermMatProd: the \"Flags\" template parameter does not match the input matrix (Eigen::ColMajor/Eigen::RowMajor)");
    }

    Index rows() const { return m_mat.rows(); }
    Index cols() const { return m_mat.cols(); }

    void perform_op(const Scalar* x_in, Scalar* y_out) const
    {
        MapConstVec x(x_in, m_mat.cols());
        MapVec y(y_out, m_mat.rows());

        y.noalias() = m_mat * x;
    }
};

std::vector<std::tuple<size_t, size_t, size_t>> SquareKagomeLattice(int64_t Nx, int64_t Ny)
{
    size_t Nbond = 22 * Nx * Ny;
    
    std::vector<std::tuple<size_t, size_t, size_t>> latt;
    latt.reserve(Nbond);

    auto mod = [](int64_t i, int64_t N)
    {
        return (i < 0) ? (i % N + N) % N : i % N;
    };
    
    for(int64_t i = 0; i < Nx; ++i)
    {
        for(int64_t j = 0; j < Ny; ++j)
        {
            latt.emplace_back(2 + 7 * (i + j * Nx), 5 + 7 * (i + j * Nx), 0);
            latt.emplace_back(3 + 7 * (i + j * Nx), 6 + 7 * (i + j * Nx), 0);
            latt.emplace_back(5 + 7 * (i + j * Nx), 4 + 7 * (mod(i + 1, Nx) + j * Nx), 0);
            latt.emplace_back(6 + 7 * (i + j * Nx), 1 + 7 * (i + mod(j + 1, Ny) * Nx), 0);
                        
            latt.emplace_back(1 + 7 * (i + j * Nx), 2 + 7 * (i + j * Nx), 1);
            latt.emplace_back(2 + 7 * (i + j * Nx), 3 + 7 * (i + j * Nx), 1);
            latt.emplace_back(3 + 7 * (i + j * Nx), 4 + 7 * (i + j * Nx), 1);
            latt.emplace_back(4 + 7 * (i + j * Nx), 1 + 7 * (i + j * Nx), 1);
                        
            latt.emplace_back(3 + 7 * (i + j * Nx), 5 + 7 * (i + j * Nx), 2);
            latt.emplace_back(4 + 7 * (i + j * Nx), 6 + 7 * (i + j * Nx), 2);
            latt.emplace_back(5 + 7 * (i + j * Nx), 1 + 7 * (mod(i + 1, Nx) + j * Nx), 2);
            latt.emplace_back(6 + 7 * (i + j * Nx), 2 + 7 * (i + mod(j + 1, Ny) * Nx), 2);
                        
            latt.emplace_back(1 + 7 * (i + j * Nx), 3 + 7 * (i + j * Nx), 3);
            latt.emplace_back(2 + 7 * (i + j * Nx), 4 + 7 * (i + j * Nx), 3);
                        
            latt.emplace_back(1 + 7 * (i + j * Nx), 7 + 7 * (i + j * Nx), 4);
            latt.emplace_back(2 + 7 * (i + j * Nx), 7 + 7 * (i + j * Nx), 4);
            latt.emplace_back(3 + 7 * (i + j * Nx), 7 + 7 * (i + j * Nx), 4);
            latt.emplace_back(4 + 7 * (i + j * Nx), 7 + 7 * (i + j * Nx), 4);
                        
            latt.emplace_back(5 + 7 * (i + j * Nx), 6 + 7 * (i + j * Nx), 5);
            latt.emplace_back(5 + 7 * (i + j * Nx), 6 + 7 * (mod(i + 1, Nx) + j * Nx), 5);
            latt.emplace_back(6 + 7 * (i + j * Nx), 5 + 7 * (i + mod(j + 1, Ny) * Nx), 5);
            latt.emplace_back(6 + 7 * (i + j * Nx), 5 + 7 * (mod(i - 1, Nx) + mod(j + 1, Ny) * Nx), 5);
        } 
    }
    
    if(latt.size() != Nbond) std::cerr << "SquareKagomeLattice: Wrong number of bonds." << std::endl;
    
    return latt;
}

int main(int argc, char* argv[]) 
{    
    omp_set_num_threads(20);
    Eigen::setNbThreads(20);
    
    const int64_t N = 14;
    const int64_t DIM = 1 << N;

    const double Junit = 170.0;
    const double gmuB = 2.0 / 0.086 * 0.05788;

    const std::vector<double> Jlist = {0.012, 0.694, 0.971, 1.000, 0.894, 0.182};

    auto pairs = SquareKagomeLattice(2, 1);

    std::vector<Edge> Jedges;

    for(const auto& [siteA, siteB, ch] : pairs) 
    {        
        Jedges.emplace_back(siteA - 1, siteB - 1, Jlist.at(ch) * Junit, Jlist.at(ch) * Junit, Jlist.at(ch) * Junit);
    }

    std::cout << "Building Hamiltonian...\n";

    Eigen::SparseMatrix<complexType, Eigen::RowMajor, int64_t> H1z(DIM, DIM);
    Eigen::SparseMatrix<complexType, Eigen::RowMajor, int64_t> H(DIM, DIM);

    build_sparse_hamiltonian(N, DIM, Jedges, H1z, H);

    H1z.makeCompressed();
    H.makeCompressed();

    std::cout << "Hamiltonian is ready.\n" << "Evaluating eigenvalue problem...\n";

    size_t num_iters = 501;
    double dBz = 500.0 / static_cast<double>(num_iters - 1UL);

    std::vector<double> Energy_ground;
    std::vector<double> TotalMagnetization_ground;

    Energy_ground.reserve(num_iters);
    TotalMagnetization_ground.reserve(num_iters);

    for(size_t iter = 0; iter < num_iters; ++iter)
    {
        std::cout << "Iteration: " << iter + 1UL << "/" << num_iters << "\n";
        
        if(iter > 0) 
        {
            H -= complexType(dBz * gmuB, 0.0) * H1z;
        }

        CustomSparseHermMatProd<complexType, Eigen::RowMajor, int64_t> opH(H);

        Spectra::HermEigsSolver<CustomSparseHermMatProd<complexType, Eigen::RowMajor, int64_t>> eigsH(opH, 1, 10);
        eigsH.init();

        int64_t nconv = eigsH.compute(Spectra::SortRule::SmallestAlge, 1000L, 1.0E-8);

        if(nconv == 0) 
        {
            std::cerr << "\nERROR: No eigenvalues converged!\n";
        
            return 1;
        }
    
        Eigen::VectorXcd evaluesH;
        Eigen::MatrixXcd evecsH;

        if(eigsH.info() == Spectra::CompInfo::Successful)
        {
            evaluesH = eigsH.eigenvalues();
            evecsH = eigsH.eigenvectors();
        }
        else 
        {
            std::cerr << "\nERROR: Spectra got an error.\n";
        
            return 1; 
        }

        //std::cout << "Ground State Energy: " << evaluesH(0).real() << std::endl;

        Energy_ground.push_back(evaluesH(0).real());

        double total_mag = 0.0;

        for (int64_t i = 0; i < DIM; ++i) 
        {
            double prob = std::norm(evecsH(i, 0));
            double sz_total = 0;

            for (int64_t s = 0; s < N; ++s) 
            {
                sz_total += (get_bit(i, s) == 0) ? 1.0 : -1.0;
            }

            total_mag += prob * sz_total;
        }

        //std::cout << "Total Magnetization at Bz = " << dBz << " is: " << total_mag << std::endl; 

        TotalMagnetization_ground.push_back(total_mag);
    }

    std::ofstream outFile1("Energy_ground.txt");
    
    if (!outFile1.is_open()) 
    {
        std::cerr << "\nError when creating Energy_ground.txt file.\n";

        return 1;
    }

    for(size_t i = 0; i < num_iters; ++i)
    {
        outFile1 << i * dBz << "\t" << Energy_ground[i] << "\n";
    }

    outFile1.close();

    std::ofstream outFile2("TotalMagnetization_ground.txt");
    
    if (!outFile2.is_open()) 
    {
        std::cerr << "\nError when creating TotalMagnetization_ground.txt.\n";

        return 1;
    }

    for(size_t i = 0; i < num_iters; ++i)
    {
        outFile2 << i * dBz << "\t" << TotalMagnetization_ground[i] << "\n";
    }

    outFile2.close();

    std::cout << "\nDone!\n";

    return 0;
}