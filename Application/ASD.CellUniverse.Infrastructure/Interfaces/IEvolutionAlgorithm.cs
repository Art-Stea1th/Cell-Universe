namespace ASD.CellUniverse.Infrastructure.Interfaces {

    public interface IEvolutionAlgorithm {

        string Name { get; }

        uint[,] Mutate(uint[,] prev);

    }
}