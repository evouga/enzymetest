#ifndef ENZYMEWRAPPER_H
#define ENZYMEWRAPPER_H

#include <vector>
#include <Eigen/Core>
#include <Eigen/Sparse>
#include "MaterialModel.h"
#include "MeshConnectivity.h"
#include "RestState.h"
#include "ElasticShell.h"
#include "MidedgeAngleSinFormulation.h"
#include "MidedgeAngleTanFormulation.h"
#include "MidedgeAverageFormulation.h"

using namespace LibShell;

extern int enzyme_dup;
extern int enzyme_dupnoneed;
extern int enzyme_out;
extern int enzyme_const;

extern void __enzyme_autodiff(void *, ...);
extern void __enzyme_fwddiff(void *, ...);
extern double* __enzyme_todense(void *, ...) noexcept;

template<class SFF>
double elasticEnergyWrapper(
            const MeshConnectivity* mesh,
            int nverts,
            const double *curPos,
            const double *edgeDOFs,
            const MaterialModel<SFF>* mat,
            const RestState *restState)
{
    Eigen::MatrixXd curPosMat(nverts, 3);
    for(int i=0; i<nverts; i++)
    {
        for(int j=0; j<3; j++)
        {
            curPosMat(i,j) = curPos[3*i+j];
        }
    }    
    
    int nedgedofs = SFF::numExtraDOFs;
    int nedges = mesh->nEdges();
    Eigen::VectorXd edgeDOFsVec(nedges*nedgedofs);
    for(int i=0; i<nedges*nedgedofs; i++)
        edgeDOFsVec[i] = edgeDOFs[i];
    
    return ElasticShell<SFF>::elasticEnergy(
            *mesh,
            curPosMat,
            edgeDOFsVec,
            *mat,
            *restState,
            NULL, NULL);
}

template<class SFF>
void grad_elasticEnergyWrapper(
            const MeshConnectivity* mesh,
            int nverts,
            const double *curPos,
            double *dcurPos,
            const double *edgeDOFs,
            double *dedgeDOFs,
            const MaterialModel<SFF>* mat,
            const RestState *restState)
{
    __enzyme_autodiff((void *)elasticEnergyWrapper<SFF>, enzyme_const, mesh, enzyme_const, nverts, enzyme_dup, curPos, dcurPos, enzyme_dup, edgeDOFs, dedgeDOFs, enzyme_const, mat, enzyme_const, restState);
}




template <class SFF>
double elasticEnergyEnzyme(
    const MeshConnectivity& mesh,
    const Eigen::MatrixXd& curPos,
    const Eigen::VectorXd& edgeDOFs,
    const MaterialModel<SFF>& mat,
    const RestState &restState,
    Eigen::VectorXd* derivative, // positions, then thetas
    std::vector<Eigen::Triplet<double> >* hessian)
{
    double energy = ElasticShell<SFF>::elasticEnergy(mesh, curPos, edgeDOFs, mat, restState, NULL, NULL);
    if(derivative || hessian)
    {
        double *rawCurPos = new double[3 * curPos.rows()];
        double *drawCurPos = new double[3 * curPos.rows()];
        for(int i=0; i<curPos.rows(); i++)
        {
            for(int j=0; j<3; j++)
            {
                rawCurPos[3*i+j] = curPos(i,j);
            }
        }
        double *rawEdgeDOFs = new double[edgeDOFs.size()];
        double *drawEdgeDOFs = new double[edgeDOFs.size()];
        for(int i=0; i<edgeDOFs.size(); i++)
            rawEdgeDOFs[i] = edgeDOFs[i];

        grad_elasticEnergyWrapper<SFF>(&mesh, curPos.rows(), rawCurPos, drawCurPos, rawEdgeDOFs, drawEdgeDOFs, &mat, &restState);
        
        int totDOFs = 3 * curPos.rows() + edgeDOFs.size();
        
        if(derivative)
        {
            derivative->resize(totDOFs);
            for(int i=0; i<3*curPos.rows(); i++)
            {
                (*derivative)[i] = drawCurPos[i];
            }
            for(int i=0; i<edgeDOFs.size(); i++)
            {
                (*derivative)[3*curPos.rows() + i] = drawEdgeDOFs[i];
            }
        }
        
        delete[] rawCurPos;
        delete[] drawCurPos;
        delete[] rawEdgeDOFs;
        delete[] drawEdgeDOFs;
    }
    return energy;
}

template double elasticEnergyEnzyme<MidedgeAngleSinFormulation>(
    const MeshConnectivity& mesh,
    const Eigen::MatrixXd& curPos,
    const Eigen::VectorXd& edgeDOFs,
    const MaterialModel<MidedgeAngleSinFormulation>& mat,
    const RestState &restState,
    Eigen::VectorXd* derivative, // positions, then thetas
    std::vector<Eigen::Triplet<double> >* hessian);
template double elasticEnergyEnzyme<MidedgeAngleTanFormulation>(
    const MeshConnectivity& mesh,
    const Eigen::MatrixXd& curPos,
    const Eigen::VectorXd& edgeDOFs,
    const MaterialModel<MidedgeAngleTanFormulation>& mat,
    const RestState &restState,
    Eigen::VectorXd* derivative, // positions, then thetas
    std::vector<Eigen::Triplet<double> >* hessian);
    
template double elasticEnergyEnzyme<MidedgeAverageFormulation>(
    const MeshConnectivity& mesh,
    const Eigen::MatrixXd& curPos,
    const Eigen::VectorXd& edgeDOFs,
    const MaterialModel<MidedgeAverageFormulation>& mat,
    const RestState &restState,
    Eigen::VectorXd* derivative, // positions, then thetas
    std::vector<Eigen::Triplet<double> >* hessian);
    
#endif
