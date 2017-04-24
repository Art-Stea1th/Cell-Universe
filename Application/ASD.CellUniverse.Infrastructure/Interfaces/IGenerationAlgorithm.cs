namespace ASD.CellUniverse.Infrastructure.Interfaces {

    public interface IGenerationAlgorithm {

        string Name { get; }

        bool[,] GenerateNextBy(bool[,] prev);

    }
}