namespace ASD.CellUniverse.Infrastructure.Interfaces {

    public interface IMutationAlgorithm {

        string Name { get; }

        uint[,] Mutate(uint[,] prev);

    }
}