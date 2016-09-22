namespace CellUniverse.Infrastructure.Interfaces {


    public interface ICellAlgorithm {

        bool[,] NextGeneration(bool[,] prevGeneration);
    }
}