namespace ASD.CellUniverse.Infrastructure.Interfaces {

    public interface IMatrixMutator {

        string Name { get; }

        byte[,] Mutate(byte[,] prev);

    }
}