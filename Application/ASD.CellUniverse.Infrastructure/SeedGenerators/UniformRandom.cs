namespace ASD.CellUniverse.Infrastructure.SeedGenerators {

    using DataProviders;
    using Interfaces;

    public class UniformRandom : ISeedGenerator {

        public string Name => "Uniform Random";
        public override string ToString() => Name;

        public byte[,] GenerateNew(int width, int height, object parameter = null) {
            var result = new byte[width, height];

            using (var random = new UniformRandomDataProvider()) {
                for (var x = 0; x < width; ++x) {
                    for (var y = 0; y < height; ++y) {
                        result[x, y] = (byte)(random.NextByte() % 2 == 0 ? 0 : 255);
                    }
                }
            }
            return result;
        }
    }
}