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
    Edge(int u_, int v_, double Jx_, double Jy_, double Jz_) : 
         u(u_), v(v_), Jx(Jx_), Jy(Jy_), Jz(Jz_) 
    {

    };

    ~Edge() = default;
    
    int u, v;
    double Jx, Jy, Jz;
};

inline int get_bit(int state, int i) 
{ 
    return (state >> i) & 1; 
}

inline int get_flipped_state(int state, int i)
{
    return state ^ (1 << i);
}

inline int get_flipped_state(int state, int i, int j)
{
    return state ^ (1 << i) ^ (1 << j);
}

void build_sparse_hamiltonian(int N, int dim,   
                              const std::vector<Edge>& edges,
                              Eigen::SparseMatrix<complexType, Eigen::ColMajor>& H1z,
                              Eigen::SparseMatrix<complexType, Eigen::ColMajor>& H) 
{       
    for (int state = 0; state < dim; ++state) 
    {
        for (int i = 0; i < N; ++i) 
        {
            int bit_i = get_bit(state, i);
            
            //int flipped_state = get_flipped_state(state, i);

            complexType sign_i = (bit_i == 0) ? complexType(1.0, 0.0) : complexType(-1.0, 0.0);

            /* if(flipped_state < state)
            {
                H1x.coeffRef(flipped_state, state) += complexType(1.0, 0.0);

                H1y.coeffRef(flipped_state, state) += complexType(0.0, 1.0) * sign_i;
            } */

            H1z.coeffRef(state, state) += sign_i;
        }
        
        for (const auto& edge : edges) 
        {
            int i = edge.u;
            int j = edge.v;

            double Jx = edge.Jx;
            double Jy = edge.Jy;
            double Jz = edge.Jz;
            
            int bit_i = get_bit(state, i);
            int bit_j = get_bit(state, j);

            int flipped_state = get_flipped_state(state, i, j);

            complexType sign_i = (bit_i == 0) ? complexType(1.0, 0.0) : complexType(-1.0, 0.0);
            complexType sign_j = (bit_j == 0) ? complexType(1.0, 0.0) : complexType(-1.0, 0.0);
            
            if(flipped_state < state)
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
    
    const int N = 24;
    const int DIM = 1 << N;

    const double Junit = 170.0;
    const double gmuB = (1.0 / 0.086) * 0.05788 * 2.0;

    /* const double J1 = 0.012;
    const double J2 = 0.694;
    const double J3 = 0.971;
    const double J4 = 1.000;
    const double J5 = 0.894;
    const double J6 = 0.182;

    std::vector<std::pair<int, int>> J1edges = {{1,6},{2,5},{3,6},{4,12},{8,13},{9,12},{10,13},{11,5}};
    std::vector<std::pair<int, int>> J2edges = {{1,2},{2,3},{3,4},{4,1},{8,9},{9,10},{10,11},{11,8}};
    std::vector<std::pair<int, int>> J3edges = {{2,6},{1,12},{4,6},{3,5},{9,13},{8,5},{11,13},{10,12}};
    std::vector<std::pair<int, int>> J4edges = {{1,3},{2,4},{8,10},{9,11}};
    std::vector<std::pair<int, int>> J5edges = {{1,7},{2,7},{3,7},{4,7},{8,14},{9,14},{10,14},{11,14}};
    std::vector<std::pair<int, int>> J6edges = {{6,12},{6,12},{5,6},{5,6},{5,13},{5,13},{12,13},{12,13}}; */

    const double J1 = 1.0;
    const double J2 = 1.0;
    const double J3 = 1.0;

    std::vector<std::pair<int, int>> J1edges = {{1,18},{2,5},{3,6},{4,11},{7,24},{8,11},{9,12},{10,5},{13,6},{14,17},{15,18},{16,23},{19,12},{20,23},{21,24},{22,17}};
    std::vector<std::pair<int, int>> J2edges = {{1,2},{2,3},{3,4},{4,1},{7,8},{8,9},{9,10},{10,7},{13,14},{14,15},{15,16},{16,13},{19,20},{20,21},{21,22},{22,19}};
    std::vector<std::pair<int, int>> J3edges = {{2,18},{1,11},{4,6},{3,5},{8,24},{7,5},{10,12},{9,11},{14,6},{13,23},{16,18},{15,17},{20,12},{19,17},{22,24},{21,23}};

    std::vector<Edge> Jedges;

    auto add_group = [&Jedges, &Junit](const std::vector<std::pair<int, int>>& e_list, double Jx, double Jy, double Jz) 
    {
        for(auto& p : e_list) 
        {
            Jedges.emplace_back(p.first - 1, p.second - 1, Jx * Junit, Jy * Junit, Jz * Junit);
        }
    };

    add_group(J1edges, J1, J1, J1);
    add_group(J2edges, J2, J2, J2);
    add_group(J3edges, J3, J3, J3);
    /* add_group(J4edges, J4, J4, J4);
    add_group(J5edges, J5, J5, J5);
    add_group(J6edges, J6, J6, J6); */

    std::cout << "Building Hamiltonian...\n";

    Eigen::SparseMatrix<complexType, Eigen::ColMajor> H1z(DIM, DIM);
    Eigen::SparseMatrix<complexType, Eigen::ColMajor> H(DIM, DIM);

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
        std::cout << "Iteration: " << iter + 1UL << "/" << num_iters << "\r";
        
        if(iter > 0) 
        {
            H -= complexType(dBz * gmuB, 0.0) * H1z;
        }

        Spectra::SparseHermMatProd<complexType, Eigen::Upper, Eigen::ColMajor> opH(H);

        Spectra::HermEigsSolver<Spectra::SparseHermMatProd<complexType, Eigen::Upper, Eigen::ColMajor>> eigsH(opH, 1, 10);
    
        eigsH.init();
        int nconv = eigsH.compute(Spectra::SortRule::SmallestAlge);

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

        for (int i = 0; i < DIM; ++i) 
        {
            double prob = std::norm(evecsH(i, 0));
            double sz_total = 0;

            for (int s = 0; s < N; ++s) 
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