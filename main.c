#include <stdio.h>
#include <stdlib.h>
#include <math.h>
#include <time.h>

#define INPUT_SIZE 64
#define GENERATOR_HIDDEN_SIZE 128
#define DISCRIMINATOR_HIDDEN_SIZE 128
#define OUTPUT_SIZE 1 // discriminator will output 0 or 1

#define IMAGE_WIDTH 32
#define IMAGE_HEIGHT 32
#define IMAGE_CHANNELS 3
#define NUM_CLASSES 10
#define NUM_TRAIN_IMAGES 50000
#define NUM_TEST_IMAGES 10000
#define NUM_LABELS 10
#define MAX_LABEL_LENGTH 100 // whatever

#define NUM_EPOCHS 100
#define NUM_BATCHES 100

typedef struct
{
    double weights[GENERATOR_HIDDEN_SIZE][INPUT_SIZE];
    double bias[GENERATOR_HIDDEN_SIZE];
} Generator;

typedef struct
{
    double weights[DISCRIMINATOR_HIDDEN_SIZE][INPUT_SIZE];
    double bias[DISCRIMINATOR_HIDDEN_SIZE];
    double output_weights[OUTPUT_SIZE][DISCRIMINATOR_HIDDEN_SIZE];
    double output_bias[OUTPUT_SIZE];
} Discriminator;

void load_binary(const char* filename, uint8_t* buffer, const int num_bytes)
{
    FILE* file = fopen(filename, "rb");
    fread(buffer, sizeof(uint8_t), num_bytes, file);
    fclose(file);
}

void load_labels(const char* filename, char labels[NUM_LABELS][MAX_LABEL_LENGTH], const int num_labels) // specialized for CIFAR-10 dataset
{
    FILE* file = fopen(filename, "r");
    if(!file)
    {
        printf("failed to open %s\n", filename);
        exit(1);
    }

    char line[MAX_LABEL_LENGTH];
    for(int i = 0; i < num_labels; i++)
    {
        fgets(line, sizeof(line), file);
        line[strlen(line)-1] = 0;
    }

    fclose(file);
}

void initialize(Generator* generator, Discriminator* discriminator)
{
    for(int i = 0; i < GENERATOR_HIDDEN_SIZE; i++)
    {
        for(int j = 0; j < INPUT_SIZE; j++)
            generator->weights[i][j] = (double) rand() / RAND_MAX; // random value between 0 and 1

        generator->bias[i] = 0; // just initialize bias to 0
    }

    for(int i = 0; i < DISCRIMINATOR_HIDDEN_SIZE; i++)
    {
        for(int j = 0; j < INPUT_SIZE; j++)
            discriminator->weights[i][j] = (double) rand() / RAND_MAX;

        discriminator->bias[i] = 0;
    }
}

// activation function
double sigmoid(double x)
{
    return 1.0 / (1.0 + exp(-x));
}

// generator forward pass
void generate(Generator* generator, double* input, double* output)
{
    // compute the hidden layer values
    double hidden_layer[GENERATOR_HIDDEN_SIZE] = {0};

    for(int i = 0; i < GENERATOR_HIDDEN_SIZE; i++)
    {
        for(int j = 0; j < INPUT_SIZE; j++)
            hidden_layer[i] += generator->weights[i][j] * input[j];

        hidden_layer[i] += generator->bias[i];
        hidden_layer[i] = sigmoid(hidden_layer[i]);
    }

    // compute the output layer values
    for(int i = 0; i < INPUT_SIZE; i++)
    {
        output[i] = 0;

        for(int j = 0; j < GENERATOR_HIDDEN_SIZE; j++)
            output[i] += generator->output_weights[i][j] * hidden_layer[j];

        output[i] += generator->output_bias[i];
        output[i] = sigmoid(output[i]);
    }
}

// discriminator forward pass
double discriminate(Discriminator* discriminator, double* input)
{
    double hidden_layer[DISCRIMINATOR_HIDDEN_SIZE] = {0};

    for(int i = 0; i < DISCRIMINATOR_HIDDEN_SIZE; i++)
    {
        for(int j = 0; j < INPUT_SIZE; j++)
            hidden_layer[i] += discriminator->weights[i][j] * input[j];

        hidden_layer[i] += discriminator->bias[i];
        hidden_layer[i] = sigmoid(hidden_layer[i]);
    }

    double output = 0;

    for(int i = 0; i < OUTPUT_SIZE; i++)
    {
        for(int j = 0; j < DISCRIMINATOR_HIDDEN_SIZE; j++)
            output += discriminator->output_weights[i][j] * hidden_layer[j];

        output += discriminator->output_bias[i];
    }

    output = sigmoid(output);

    return output;
}

void train(Generator* generator, Discriminator* discriminator, double* real_data)
{
    double noise[INPUT_SIZE];
    for(int i = 0; i < INPUT_SIZE; i++)
        noise[i] = rand();

    // generate fake data
    double fake_data[INPUT_SIZE];
    generate(generator, noise, fake_data);

    // train the discriminator
    double real_output = discriminate(discriminator, real_data);
    double fake_output = discriminate(discriminator, fake_data);

    // update discriminator weights and biases based on the loss

    // train the generator
    generate(generator, noise, fake_data);
    double generator_output = discriminate(discriminator, fake_data);

    // update generator weights and biases based on the loss
}

int main()
{
    // initialize
    srand(time(NULL));
    Generator generator;
    Discriminator discriminator;
    initialize(&generator, &discriminator);

    double noise[INPUT_SIZE];
    for(int i = 0; i < INPUT_SIZE; i++)
        noise[i] = rand();

    uint8_t train_images[NUM_TRAIN_IMAGES * IMAGE_WIDTH * IMAGE_HEIGHT * IMAGE_CHANNELS];
    uint8_t train_labels[NUM_TRAIN_IMAGES];
    load_binary("dataset/cifar-10-binary/cifar-10-batches-bin/data_batch_1.bin", train_images, NUM_TRAIN_IMAGES * IMAGE_WIDTH * IMAGE_HEIGHT * IMAGE_CHANNELS);
    load_binary("dataset/cifar-10-binary/cifar-10-batches-bin/batches.meta.txt", train_labels, NUM_LABELS);
    // training loop
    for(int epoch = 0; epoch < NUM_EPOCHS, epoch++)
    {
        for(int i = 0; i < NUM_BATCHES; i++)
        {
            // load a batch of real data
            double real_data[INPUT_SIZE];

            // train
            train(&generator, &discriminator, real_data, noise);
        }
    }

    double noise[INPUT_SIZE];
    for(int i = 0; i < INPUT_SIZE; i++)
        noise[i] = rand();

    double fake_image[INPUT_SIZE];
    generate(&generator, noise, fake_image);

    // output the generated image

    return 0;
}