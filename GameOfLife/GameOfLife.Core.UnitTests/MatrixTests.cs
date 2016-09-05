using NUnit.Framework;


namespace GameOfLife.Core.UnitTests.Core.UnitTests {

    [TestFixture]
    class MatrixTests {

        private Matrix<bool> matrix;

        [SetUp]
        public void Setup() { matrix = new Matrix<bool>(); }
        [TearDown]
        public void Teardown() { matrix = null; }

        // - Rows & Columns compliance

        [Test]
        public void RowsAndColumnsCompliance_FactoryMethodAndIndexer_AreEqual() {

            int rows = 3, columns = 5;
            matrix = new Matrix<bool>(rows, columns);

            bool result = matrix[rows - 1, columns - 1] = true;

            Assert.IsTrue(result);
        }

        [Test]
        public void RowsAndColumnsCompliance_FactoryMethodAndProperties_AreEqual() {

            int rows = 3, columns = 5;
            matrix = new Matrix<bool>(rows, columns);

            Assert.AreEqual(rows, matrix.RowsCount);
            Assert.AreEqual(columns, matrix.ColumnsCount);
        }

        // - Matrix<T> CreateNew() Tests

        [Test]
        public void Ctor_AnyArgs_ReturnNotNull() {
            for (int row = -1; row <= 1; row++) {
                for (int column = -1; column <= 1; column++) {
                    Assert.IsNotNull(new Matrix<bool>(row, column));
                }
            }
        }

        // - bool IsValidSize() Tests

        [Test]
        public void IsValidSize_PositiveArgs_ReturnTrue() {
            Assert.IsTrue(Matrix<bool>.IsValidSize(1, 1));
        }

        [Test]
        public void IsValidSize_NegativeOrNullArgs_ReturnFalse() {
            for (int row = -1; row <= 0; row++) {
                for (int column = -1; column <= 0; column++) {
                    Assert.IsFalse(Matrix<bool>.IsValidSize(row, column));
                }
            }
        }
    }
}