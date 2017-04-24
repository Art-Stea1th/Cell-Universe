namespace ASD.CellUniverse.Infrastructure.Interfaces {

    public interface ISeedGenerator { // Future features: ISeed, IGenerationMode

        bool[,] Generate(int width, int height);

    }
}