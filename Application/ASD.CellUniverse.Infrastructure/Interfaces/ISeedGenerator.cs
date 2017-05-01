namespace ASD.CellUniverse.Infrastructure.Interfaces {

    public interface ISeedGenerator {

        string Name { get; }

        uint[,] GenerateNew(int width, int height, object parameter = null);

    }
}