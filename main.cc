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
#include <Spectra/MatOp/SparseHermMatProd.h>

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
                              Eigen::SparseMatrix<complexType, Eigen::ColMajor, int64_t>& H1z,
                              Eigen::SparseMatrix<complexType, Eigen::ColMajor, int64_t>& H) 
{       
    for (int64_t state = 0; state < dim; ++state) 
    {
        for (int64_t i = 0; i < N; ++i) 
        {
            int64_t bit_i = get_bit(state, i);
            
            //int64_t flipped_state = get_flipped_state(state, i);

            complexType sign_i = (bit_i == 0) ? complexType(1.0, 0.0) : complexType(-1.0, 0.0);

            /* if(flipped_state > state)
            {
                H1x.coeffRef(flipped_state, state) += complexType(1.0, 0.0);

                H1y.coeffRef(flipped_state, state) += complexType(0.0, 1.0) * sign_i;
            } */

            H1z.coeffRef(state, state) += sign_i;
        }
        
        for (const auto& edge : edges) 
        {
            int64_t i = edge.u;
            int64_t j = edge.v;

            double Jx = edge.Jx;
            double Jy = edge.Jy;
            double Jz = edge.Jz;
            
            int64_t bit_i = get_bit(state, i);
            int64_t bit_j = get_bit(state, j);

            int64_t flipped_state = get_flipped_state(state, i, j);

            complexType sign_i = (bit_i == 0) ? complexType(1.0, 0.0) : complexType(-1.0, 0.0);
            complexType sign_j = (bit_j == 0) ? complexType(1.0, 0.0) : complexType(-1.0, 0.0);
            
            if(flipped_state > state)
            {
                H.coeffRef(flipped_state, state) += complexType(Jx, 0.0);

                H.coeffRef(flipped_state, state) += complexType(-Jy, 0.0) * sign_i * sign_j;
            }
            
            H.coeffRef(state, state) += complexType(Jz, 0.0) * sign_i * sign_j;
        }
    }
}

int main(int argc, char* argv[]) 
{    
    Eigen::initParallel();
    omp_set_num_threads(20);
    Eigen::setNbThreads(20);
    
    const int64_t N = 28;
    const int64_t DIM = 1 << N;

    const double Junit = 170.0;
    const double gmuB = (1.0 / 0.086) * 0.05788 * 2.0;

    const double J1 = 0.012;
    const double J2 = 0.694;
    const double J3 = 0.971;
    const double J4 = 1.000;
    const double J5 = 0.894;
    const double J6 = 0.182;

    /* std::vector<std::pair<int64_t, int64_t>> J1edges = {{1,6},{2,5},{3,6},{4,12},{8,13},{9,12},{10,13},{11,5}};
    std::vector<std::pair<int64_t, int64_t>> J2edges = {{1,2},{2,3},{3,4},{4,1},{8,9},{9,10},{10,11},{11,8}};
    std::vector<std::pair<int64_t, int64_t>> J3edges = {{2,6},{1,12},{4,6},{3,5},{9,13},{8,5},{11,13},{10,12}};
    std::vector<std::pair<int64_t, int64_t>> J4edges = {{1,3},{2,4},{8,10},{9,11}};
    std::vector<std::pair<int64_t, int64_t>> J5edges = {{1,7},{2,7},{3,7},{4,7},{8,14},{9,14},{10,14},{11,14}};
    std::vector<std::pair<int64_t, int64_t>> J6edges = {{6,12},{6,12},{5,6},{5,6},{5,13},{5,13},{12,13},{12,13}}; */

    std::vector<std::pair<int64_t, int64_t>> J1edges = {{1,20},{2,5},{3,6},{4,12},{8,27},{9,12},{10,13},{11,5},{15,6},{16,19},{17,20},{18,26},{22,13},{23,26},{24,27},{25,19}};
    std::vector<std::pair<int64_t, int64_t>> J2edges = {{1,2},{2,3},{3,4},{4,1},{8,9},{9,10},{10,11},{11,8},{15,16},{16,17},{17,18}, {18,15},{22,23},{23,24},{24,25},{25,22}};
    std::vector<std::pair<int64_t, int64_t>> J3edges = {{2,20},{1,12},{4,6},{3,5},{9,27},{8,5},{11,13},{10,12},{16,6},{15,26},{18,20},{17,19},{23,13},{22,19},{25,27},{24,26}};
    std::vector<std::pair<int64_t, int64_t>> J4edges = {{1,3},{2,4},{8,10},{9,11},{15,17},{16,18},{22,24},{23,25}};
    std::vector<std::pair<int64_t, int64_t>> J5edges = {{1,7},{2,7},{3,7},{4,7},{8,14},{9,14},{10,14},{11,14},{15,21},{16,21},{17,21},{18,21},{22,28},{23,28},{24,28},{25,28}};
    std::vector<std::pair<int64_t, int64_t>> J6edges = {{6,12},{12,20},{20,5},{5,6},{13,5},{5,27},{27,12},{12,13},{20,26},{26,6},{6,19},{19,20},{27,19},{19,13},{13,26},{26,27}};

    std::vector<Edge> Jedges;

    auto add_group = [&Jedges, &Junit](const std::vector<std::pair<int64_t, int64_t>>& e_list, double Jx, double Jy, double Jz) 
    {
        for(auto& p : e_list) 
        {
            Jedges.emplace_back(p.first - 1, p.second - 1, Jx * Junit, Jy * Junit, Jz * Junit);
        }
    };

    add_group(J1edges, J1, J1, J1);
    add_group(J2edges, J2, J2, J2);
    add_group(J3edges, J3, J3, J3);
    add_group(J4edges, J4, J4, J4);
    add_group(J5edges, J5, J5, J5);
    add_group(J6edges, J6, J6, J6);

    std::cout << "Building Hamiltonian...\n";

    Eigen::SparseMatrix<complexType, Eigen::ColMajor, int64_t> H1z(DIM, DIM);
    Eigen::SparseMatrix<complexType, Eigen::ColMajor, int64_t> H(DIM, DIM);

    build_sparse_hamiltonian(N, DIM, Jedges, H1z, H);

    H1z.makeCompressed();
    H.makeCompressed();

    std::cout << "Hamiltonian is ready.\n" << "Evaluating eigenvalue problem...\n";

    size_t num_iters = 401;
    double dBz = 400.0 / static_cast<double>(num_iters - 1UL);

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

        Spectra::SparseHermMatProd<complexType, Eigen::Lower, Eigen::ColMajor, int64_t> opH(H);

        Spectra::HermEigsSolver<Spectra::SparseHermMatProd<complexType, Eigen::Lower, Eigen::ColMajor, int64_t>> eigsH(opH, 1, 10);
    
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