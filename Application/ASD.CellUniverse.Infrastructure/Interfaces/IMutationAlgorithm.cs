namespace ASD.CellUniverse.Infrastructure.Interfaces {

    public interface IMutationAlgorithm {

        string Name { get; }

        byte[,] Mutate(byte[,] prev);

    }
}