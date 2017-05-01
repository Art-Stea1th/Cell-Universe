namespace ASD.CellUniverse.Infrastructure.SeedGenerators {

    using DataProviders;
    using Interfaces;

    public class UniformRandom : ISeedGenerator {

        public string Name => "Uniform Random";
        public override string ToString() => Name;

        public uint[,] GenerateNew(int width, int height, object parameter = null) {
            var result = new uint[width, height];

            using (var random = new UniformRandomDataProvider()) {
                for (var x = 0; x < width; ++x) {
                    for (var y = 0; y < height; ++y) {
                        result[x, y] = random.NextByte() % 2 == 0 ? 0 : (uint)255 << 24;
                        //result[x, y] = random.NextUint();
                    }
                }
            }
            return result;
        }
    }
}